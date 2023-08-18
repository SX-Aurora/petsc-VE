#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#ifdef PETSC_HAVE_EGADS
#include <egads.h>
#endif

#include <ctetgen.h>

/* This is to fix the tetrahedron orientation from TetGen */
static PetscErrorCode DMPlexInvertCells_CTetgen(PetscInt numCells, PetscInt numCorners, PetscInt cells[])
{
  PetscInt bound = numCells*numCorners, coff;

  PetscFunctionBegin;
#define SWAP(a,b) do { PetscInt tmp = (a); (a) = (b); (b) = tmp; } while (0)
  for (coff = 0; coff < bound; coff += numCorners) SWAP(cells[coff],cells[coff+1]);
#undef SWAP
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexGenerate_CTetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm               comm;
  const PetscInt         dim = 3;
  PLC                   *in, *out;
  DMUniversalLabel       universal;
  PetscInt               vStart, vEnd, v, eStart, eEnd, e, fStart, fEnd, f, verbose = 0;
  DMPlexInterpolatedFlag isInterpolated;
  PetscMPIInt            rank;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(NULL,((PetscObject) boundary)->prefix, "-ctetgen_verbose", &verbose, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)boundary,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = DMPlexIsInterpolatedCollective(boundary, &isInterpolated);CHKERRQ(ierr);
  ierr = DMUniversalLabelCreate(boundary, &universal);CHKERRQ(ierr);

  ierr = PLCCreate(&in);CHKERRQ(ierr);
  ierr = PLCCreate(&out);CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscSection       coordSection;
    Vec                coordinates;
    const PetscScalar *array;

    ierr = PetscMalloc1(in->numberofpoints*dim, &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberofpoints,     &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      ierr = DMLabelGetValue(universal->label, v, &m);CHKERRQ(ierr);
      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArrayRead(coordinates, &array);CHKERRQ(ierr);
  }

  ierr = DMPlexGetHeightStratum(boundary, 1, &eStart, &eEnd);CHKERRQ(ierr);
  in->numberofedges = eEnd - eStart;
  if (isInterpolated == DMPLEX_INTERPOLATED_FULL && in->numberofedges > 0) {
    ierr = PetscMalloc1(in->numberofedges*2, &in->edgelist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberofedges,   &in->edgemarkerlist);CHKERRQ(ierr);
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;
      PetscInt        coneSize, val;

      ierr = DMPlexGetConeSize(boundary, e, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(boundary, e, &cone);CHKERRQ(ierr);
      in->edgelist[idx*2]     = cone[0] - vStart;
      in->edgelist[idx*2 + 1] = cone[1] - vStart;

      ierr = DMLabelGetValue(universal->label, e, &val);CHKERRQ(ierr);
      in->edgemarkerlist[idx] = (int) val;
    }
  }

  ierr = DMPlexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);
  in->numberoffacets = fEnd - fStart;
  if (in->numberoffacets > 0) {
    ierr = PetscMalloc1(in->numberoffacets, &in->facetlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberoffacets, &in->facetmarkerlist);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx    = f - fStart;
      PetscInt      *points = NULL, numPoints, p, numVertices = 0, v, m = -1;
      polygon       *poly;

      in->facetlist[idx].numberofpolygons = 1;
      ierr = PetscMalloc1(in->facetlist[idx].numberofpolygons, &in->facetlist[idx].polygonlist);CHKERRQ(ierr);
      in->facetlist[idx].numberofholes    = 0;
      in->facetlist[idx].holelist         = NULL;

      ierr = DMPlexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
      }

      poly                   = in->facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      ierr = PetscMalloc1(poly->numberofvertices, &poly->vertexlist);CHKERRQ(ierr);
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      ierr = DMLabelGetValue(universal->label, f, &m);CHKERRQ(ierr);
      in->facetmarkerlist[idx] = (int) m;
      ierr = DMPlexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    }
  }
  if (rank == 0) {
    TetGenOpts t;

    ierr        = TetGenOptsInitialize(&t);CHKERRQ(ierr);
    t.in        = boundary; /* Should go away */
    t.plc       = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose;
#if 0
#ifdef PETSC_HAVE_EGADS
    /* Need to add in more TetGen code */
    t.nobisect  = 1; /* Add Y to preserve Surface Mesh for EGADS */
#endif
#endif

    ierr        = TetGenCheckOpts(&t);CHKERRQ(ierr);
    ierr        = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    PetscReal      *meshCoords = NULL;
    PetscInt       *cells      = NULL;

    if (sizeof (PetscReal) == sizeof (out->pointlist[0])) {
      meshCoords = (PetscReal *) out->pointlist;
    } else {
      PetscInt i;

      ierr = PetscMalloc1(dim * numVertices, &meshCoords);CHKERRQ(ierr);
      for (i = 0; i < dim * numVertices; ++i) meshCoords[i] = (PetscReal) out->pointlist[i];
    }
    if (sizeof (PetscInt) == sizeof (out->tetrahedronlist[0])) {
      cells = (PetscInt *) out->tetrahedronlist;
    } else {
      PetscInt i;

      ierr = PetscMalloc1(numCells * numCorners, &cells);CHKERRQ(ierr);
      for (i = 0; i < numCells * numCorners; i++) cells[i] = (PetscInt) out->tetrahedronlist[i];
    }

    ierr = DMPlexInvertCells_CTetgen(numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    if (sizeof (PetscReal) != sizeof (out->pointlist[0])) {
      ierr = PetscFree(meshCoords);CHKERRQ(ierr);
    }
    if (sizeof (PetscInt) != sizeof (out->tetrahedronlist[0])) {
      ierr = PetscFree(cells);CHKERRQ(ierr);
    }

    /* Set labels */
    ierr = DMUniversalLabelCreateLabels(universal, PETSC_TRUE, *dm);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMUniversalLabelSetLabelValue(universal, *dm, PETSC_TRUE, v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMUniversalLabelSetLabelValue(universal, *dm, PETSC_TRUE, edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMUniversalLabelSetLabelValue(universal, *dm, PETSC_TRUE, faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }

#ifdef PETSC_HAVE_EGADS
    {
      DMLabel        bodyLabel;
      PetscContainer modelObj;
      PetscInt       cStart, cEnd, c, eStart, eEnd, fStart, fEnd;
      ego           *bodies;
      ego            model, geom;
      int            Nb, oclass, mtype, *senses;

      /* Get Attached EGADS Model from Original DMPlex */
      ierr = PetscObjectQuery((PetscObject) boundary, "EGADS Model", (PetscObject *) &modelObj);CHKERRQ(ierr);
      if (modelObj) {
        ierr = PetscContainerGetPointer(modelObj, (void **) &model);CHKERRQ(ierr);
        ierr = EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses);CHKERRQ(ierr);
        /* Transfer EGADS Model to Volumetric Mesh */
        ierr = PetscObjectCompose((PetscObject) *dm, "EGADS Model", (PetscObject) modelObj);CHKERRQ(ierr);

        /* Set Cell Labels */
        ierr = DMGetLabel(*dm, "EGADS Body ID", &bodyLabel);CHKERRQ(ierr);
        ierr = DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
        ierr = DMPlexGetHeightStratum(*dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(*dm, 1, &eStart, &eEnd);CHKERRQ(ierr);

        for (c = cStart; c < cEnd; ++c) {
          PetscReal centroid[3] = {0., 0., 0.};
          PetscInt  b;

          /* Deterimine what body the cell's centroid is located in */
          if (!interpolate) {
            PetscSection   coordSection;
            Vec            coordinates;
            PetscScalar   *coords = NULL;
            PetscInt       coordSize, s, d;

            ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
            ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
            ierr = DMPlexVecGetClosure(*dm, coordSection, coordinates, c, &coordSize, &coords);CHKERRQ(ierr);
            for (s = 0; s < coordSize; ++s) for (d = 0; d < dim; ++d) centroid[d] += coords[s*dim+d];
            ierr = DMPlexVecRestoreClosure(*dm, coordSection, coordinates, c, &coordSize, &coords);CHKERRQ(ierr);
          } else {
            ierr = DMPlexComputeCellGeometryFVM(*dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
          }
          for (b = 0; b < Nb; ++b) {
            if (EG_inTopology(bodies[b], centroid) == EGADS_SUCCESS) break;
          }
          if (b < Nb) {
            PetscInt   cval = b, eVal, fVal;
            PetscInt *closure = NULL, Ncl, cl;

            ierr = DMLabelSetValue(bodyLabel, c, cval);CHKERRQ(ierr);
            ierr = DMPlexGetTransitiveClosure(*dm, c, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
            for (cl = 0; cl < Ncl; ++cl) {
              const PetscInt p = closure[cl*2];

              if (p >= eStart && p < eEnd) {
                ierr = DMLabelGetValue(bodyLabel, p, &eVal);CHKERRQ(ierr);
                if (eVal < 0) {ierr = DMLabelSetValue(bodyLabel, p, cval);CHKERRQ(ierr);}
              }
              if (p >= fStart && p < fEnd) {
                ierr = DMLabelGetValue(bodyLabel, p, &fVal);CHKERRQ(ierr);
                if (fVal < 0) {ierr = DMLabelSetValue(bodyLabel, p, cval);CHKERRQ(ierr);}
              }
            }
            ierr = DMPlexRestoreTransitiveClosure(*dm, c, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
          }
        }
      }
    }
#endif
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = DMUniversalLabelDestroy(&universal);CHKERRQ(ierr);
  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexRefine_CTetgen(DM dm, PetscReal *maxVolumes, DM *dmRefined)
{
  MPI_Comm               comm;
  const PetscInt         dim = 3;
  PLC                   *in, *out;
  DMUniversalLabel       universal;
  PetscInt               vStart, vEnd, v, eStart, eEnd, e, fStart, fEnd, f, cStart, cEnd, c, verbose = 0;
  DMPlexInterpolatedFlag isInterpolated;
  PetscMPIInt            rank;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(NULL,((PetscObject) dm)->prefix, "-ctetgen_verbose", &verbose, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = DMPlexIsInterpolatedCollective(dm, &isInterpolated);CHKERRQ(ierr);
  ierr = DMUniversalLabelCreate(dm, &universal);CHKERRQ(ierr);

  ierr = PLCCreate(&in);CHKERRQ(ierr);
  ierr = PLCCreate(&out);CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc1(in->numberofpoints*dim, &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberofpoints, &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      ierr = DMLabelGetValue(universal->label, v, &m);CHKERRQ(ierr);
      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }

  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  in->numberofedges = eEnd - eStart;
  if (isInterpolated == DMPLEX_INTERPOLATED_FULL && in->numberofedges > 0) {
    ierr = PetscMalloc1(in->numberofedges * 2, &in->edgelist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberofedges,     &in->edgemarkerlist);CHKERRQ(ierr);
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;
      PetscInt        coneSize, val;

      ierr = DMPlexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      in->edgelist[idx*2]     = cone[0] - vStart;
      in->edgelist[idx*2 + 1] = cone[1] - vStart;

      ierr = DMLabelGetValue(universal->label, e, &val);CHKERRQ(ierr);
      in->edgemarkerlist[idx] = (int) val;
    }
  }

  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  in->numberoftrifaces = 0;
  for (f = fStart; f < fEnd; ++f) {
    PetscInt supportSize;

    ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
    if (supportSize == 1) ++in->numberoftrifaces;
  }
  if (isInterpolated == DMPLEX_INTERPOLATED_FULL && in->numberoftrifaces > 0) {
    PetscInt tf = 0;

    ierr = PetscMalloc1(in->numberoftrifaces*3, &in->trifacelist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberoftrifaces, &in->trifacemarkerlist);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      PetscInt *points = NULL;
      PetscInt supportSize, numPoints, p, Nv = 0, val;

      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      if (supportSize != 1) continue;
      ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) in->trifacelist[tf*3 + Nv++] = point - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      if (Nv != 3) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D has %D vertices, not 3", f, Nv);
      ierr = DMLabelGetValue(universal->label, f, &val);CHKERRQ(ierr);
      in->trifacemarkerlist[tf] = (int) val;
      ++tf;
    }
  }

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  in->numberofcorners       = 4;
  in->numberoftetrahedra    = cEnd - cStart;
  in->tetrahedronvolumelist = maxVolumes;
  if (in->numberoftetrahedra > 0) {
    ierr = PetscMalloc1(in->numberoftetrahedra*in->numberofcorners, &in->tetrahedronlist);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) in->tetrahedronlist[idx*in->numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }

  if (rank == 0) {
    TetGenOpts t;

    ierr = TetGenOptsInitialize(&t);CHKERRQ(ierr);

    t.in        = dm; /* Should go away */
    t.refine    = 1;
    t.varvolume = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose; /* Change this */

    ierr = TetGenCheckOpts(&t);CHKERRQ(ierr);
    ierr = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }

  in->tetrahedronvolumelist = NULL;
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    PetscReal      *meshCoords = NULL;
    PetscInt       *cells      = NULL;
    PetscBool      interpolate = isInterpolated == DMPLEX_INTERPOLATED_FULL ? PETSC_TRUE : PETSC_FALSE;

    if (sizeof (PetscReal) == sizeof (out->pointlist[0])) {
      meshCoords = (PetscReal *) out->pointlist;
    } else {
      PetscInt i;

      ierr = PetscMalloc1(dim * numVertices, &meshCoords);CHKERRQ(ierr);
      for (i = 0; i < dim * numVertices; ++i) meshCoords[i] = (PetscReal) out->pointlist[i];
    }
    if (sizeof (PetscInt) == sizeof (out->tetrahedronlist[0])) {
      cells = (PetscInt *) out->tetrahedronlist;
    } else {
      PetscInt i;

      ierr = PetscMalloc1(numCells * numCorners, &cells);CHKERRQ(ierr);
      for (i = 0; i < numCells * numCorners; ++i) cells[i] = (PetscInt) out->tetrahedronlist[i];
    }

    ierr = DMPlexInvertCells_CTetgen(numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    if (sizeof (PetscReal) != sizeof (out->pointlist[0])) {ierr = PetscFree(meshCoords);CHKERRQ(ierr);}
    if (sizeof (PetscInt) != sizeof (out->tetrahedronlist[0])) {ierr = PetscFree(cells);CHKERRQ(ierr);}

    /* Set labels */
    ierr = DMUniversalLabelCreateLabels(universal, PETSC_TRUE, *dmRefined);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        ierr = DMUniversalLabelSetLabelValue(universal, *dmRefined, PETSC_TRUE, v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);
      }
    }
    if (interpolate) {
      PetscInt e, f;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          ierr = DMUniversalLabelSetLabelValue(universal, *dmRefined, PETSC_TRUE, edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          ierr = DMUniversalLabelSetLabelValue(universal, *dmRefined, PETSC_TRUE, faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);
          ierr = DMPlexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }

#ifdef PETSC_HAVE_EGADS
    {
      DMLabel        bodyLabel;
      PetscContainer modelObj;
      PetscInt       cStart, cEnd, c, eStart, eEnd, fStart, fEnd;
      ego           *bodies;
      ego            model, geom;
      int            Nb, oclass, mtype, *senses;

      /* Get Attached EGADS Model from Original DMPlex */
      ierr = PetscObjectQuery((PetscObject) dm, "EGADS Model", (PetscObject *) &modelObj);CHKERRQ(ierr);
      if (modelObj) {
        ierr = PetscContainerGetPointer(modelObj, (void **) &model);CHKERRQ(ierr);
        ierr = EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses);CHKERRQ(ierr);
        /* Transfer EGADS Model to Volumetric Mesh */
        ierr = PetscObjectCompose((PetscObject) *dmRefined, "EGADS Model", (PetscObject) modelObj);CHKERRQ(ierr);

        /* Set Cell Labels */
        ierr = DMGetLabel(*dmRefined, "EGADS Body ID", &bodyLabel);CHKERRQ(ierr);
        ierr = DMPlexGetHeightStratum(*dmRefined, 0, &cStart, &cEnd);CHKERRQ(ierr);
        ierr = DMPlexGetHeightStratum(*dmRefined, 1, &fStart, &fEnd);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(*dmRefined, 1, &eStart, &eEnd);CHKERRQ(ierr);

        for (c = cStart; c < cEnd; ++c) {
          PetscReal centroid[3] = {0., 0., 0.};
          PetscInt  b;

          /* Deterimine what body the cell's centroid is located in */
          if (!interpolate) {
            PetscSection   coordSection;
            Vec            coordinates;
            PetscScalar   *coords = NULL;
            PetscInt       coordSize, s, d;

            ierr = DMGetCoordinatesLocal(*dmRefined, &coordinates);CHKERRQ(ierr);
            ierr = DMGetCoordinateSection(*dmRefined, &coordSection);CHKERRQ(ierr);
            ierr = DMPlexVecGetClosure(*dmRefined, coordSection, coordinates, c, &coordSize, &coords);CHKERRQ(ierr);
            for (s = 0; s < coordSize; ++s) for (d = 0; d < dim; ++d) centroid[d] += coords[s*dim+d];
            ierr = DMPlexVecRestoreClosure(*dmRefined, coordSection, coordinates, c, &coordSize, &coords);CHKERRQ(ierr);
          } else {
            ierr = DMPlexComputeCellGeometryFVM(*dmRefined, c, NULL, centroid, NULL);CHKERRQ(ierr);
          }
          for (b = 0; b < Nb; ++b) {
            if (EG_inTopology(bodies[b], centroid) == EGADS_SUCCESS) break;
          }
          if (b < Nb) {
            PetscInt   cval = b, eVal, fVal;
            PetscInt *closure = NULL, Ncl, cl;

            ierr = DMLabelSetValue(bodyLabel, c, cval);CHKERRQ(ierr);
            ierr = DMPlexGetTransitiveClosure(*dmRefined, c, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
            for (cl = 0; cl < Ncl; cl += 2) {
              const PetscInt p = closure[cl];

              if (p >= eStart && p < eEnd) {
                ierr = DMLabelGetValue(bodyLabel, p, &eVal);CHKERRQ(ierr);
                if (eVal < 0) {ierr = DMLabelSetValue(bodyLabel, p, cval);CHKERRQ(ierr);}
              }
              if (p >= fStart && p < fEnd) {
                ierr = DMLabelGetValue(bodyLabel, p, &fVal);CHKERRQ(ierr);
                if (fVal < 0) {ierr = DMLabelSetValue(bodyLabel, p, cval);CHKERRQ(ierr);}
              }
            }
            ierr = DMPlexRestoreTransitiveClosure(*dmRefined, c, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
          }
        }
      }
    }
#endif
    ierr = DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = DMUniversalLabelDestroy(&universal);CHKERRQ(ierr);
  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
