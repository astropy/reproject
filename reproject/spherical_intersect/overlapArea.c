/* Methods to compute pixel overlap areas on the sphere.
 *
 * Originally developed in 2003 / 2004 by John Good.
 */

#include <stdio.h>
#if defined(_MSC_VER)
  #define _USE_MATH_DEFINES
#endif
#include <math.h>
#include "mNaN.h"
#include "overlapArea.h"

// Constants

#define FALSE             0
#define TRUE              1
#define FOREVER           1

#define COLINEAR_SEGMENTS 0
#define ENDPOINT_ONLY     1
#define NORMAL_INTERSECT  2
#define NO_INTERSECTION   3

#define CLOCKWISE         1
#define PARALLEL          0
#define COUNTERCLOCKWISE -1

#define UNKNOWN           0
#define P_IN_Q            1
#define Q_IN_P            2

#if defined(DEBUG_OVERLAP_AREA)
    const int DEBUG = 10;
#else
    const int DEBUG = 0;
#endif

const double DEG_TO_RADIANS = M_PI / 180.;
// sin(x) where x = 5e-4 arcsec or cos(x) when x is within 1e-5 arcsec of 90 degrees
const double TOLERANCE = 4.424e-9;
const int NP = 4;
const int NQ = 4;

typedef struct vec {
  double x;
  double y;
  double z;
} Vec;

// Function prototypes

int DirectionCalculator(Vec *a, Vec *b, Vec *c);
int SegSegIntersect(Vec *a, Vec *b, Vec *c, Vec *d, Vec *e, Vec *f, Vec *p,
                    Vec *q);
int Between(Vec *a, Vec *b, Vec *c);
int Cross(Vec *a, Vec *b, Vec *c);
double Dot(Vec *a, Vec *b);
double Normalize(Vec *a);
void Reverse(Vec *a);
void SaveVertex(Vec *a, Vec *P, Vec *Q, Vec *V, int *nv);
void SaveSharedSeg(Vec *p, Vec *q, Vec *P, Vec *Q, Vec *V, int *nv);

void ComputeIntersection(Vec *P, Vec *Q, Vec *V, int *nv);
void EnsureCounterClockWise(Vec *V);

int UpdateInteriorFlag(Vec *p, int interiorFlag, int pEndpointFromQdir,
                       int qEndpointFromPdir, Vec *P, Vec *Q, Vec *V, int *nv);

int Advance(int i, int *i_advances, int n, int inside, Vec *v, Vec *P, Vec *Q, Vec *V, int *nv);

double Girard(int nv, Vec *V);
void RemoveDups(int *nv, Vec *V);

int printDir(char *point, char *vector, int dir);

/*
 * Sets up the polygons, runs the overlap computation, and returns the area of overlap.
 */
double computeOverlap(double *ilon, double *ilat, double *olon, double *olat,
                      int energyMode, double refArea, double *areaRatio) {
  int i;
  double thisPixelArea;
  Vec P[8], Q[8], V[16];
  int nv;

  *areaRatio = 1.;

  if (energyMode) {
    nv = 0;

    for (i = 0; i < 4; ++i)
      SaveVertex(&P[i], P, Q, V, &nv);

    thisPixelArea = Girard(nv, V);

    *areaRatio = thisPixelArea / refArea;
  }

  nv = 0;

  if (DEBUG >= 4) {
    printf("Input (P):\n");
    for (i = 0; i < 4; ++i)
      printf("%10.6f %10.6f\n", ilon[i], ilat[i]);

    printf("\nOutput (Q):\n");
    for (i = 0; i < 4; ++i)
      printf("%10.6f %10.6f\n", olon[i], olat[i]);

    printf("\n");
    fflush(stdout);
  }

  for (i = 0; i < 4; ++i) {
    P[i].x = cos(ilon[i]) * cos(ilat[i]);
    P[i].y = sin(ilon[i]) * cos(ilat[i]);
    P[i].z = sin(ilat[i]);
  }

  for (i = 0; i < 4; ++i) {
    Q[i].x = cos(olon[i]) * cos(olat[i]);
    Q[i].y = sin(olon[i]) * cos(olat[i]);
    Q[i].z = sin(olat[i]);
  }

  EnsureCounterClockWise(P);
  EnsureCounterClockWise(Q);

  ComputeIntersection(P, Q, V, &nv);

  return (Girard(nv, V));
}

void EnsureCounterClockWise(Vec *V) {

  // Make sure that the polygon is counter-clockwise. For now we assume that we
  // are dealing with convex quadrilaterals, and therefore we can just check
  // the two first sides of the polygon.

  Vec S1, S2, C;
  double dir;
  double tmp;

  S1.x = V[1].x - V[0].x;
  S1.y = V[1].y - V[0].y;
  S1.z = V[1].z - V[0].z;

  S2.x = V[2].x - V[1].x;
  S2.y = V[2].y - V[1].y;
  S2.z = V[2].z - V[1].z;

  Cross(&S1, &S2, &C);

  dir = Dot(&V[1], &C);

  if (dir < 0) {
    tmp = V[2].x;
    V[2].x = V[0].x;
    V[0].x = tmp;
    tmp = V[2].y;
    V[2].y = V[0].y;
    V[0].y = tmp;
    tmp = V[2].z;
    V[2].z = V[0].z;
    V[0].z = tmp;
  }

}

/*
 * Find the polygon defining the area of overlap
 * between the two input polygons P and Q.
 */
void ComputeIntersection(Vec *P, Vec *Q, Vec *V, int *nv) {
  Vec Pdir, Qdir;              // "Current" directed edges on P and Q
  Vec other;                   // Temporary "edge-like" variable
  int ip, iq;                  // Indices of ends of Pdir, Qdir
  int ip_begin, iq_begin;      // Indices of beginning of Pdir, Qdir
  int PToQDir;                 // Qdir direction relative to Pdir
                               // (e.g. CLOCKWISE)
  int qEndpointFromPdir;       // End P vertex as viewed from beginning
                               // of Qdir relative to Qdir
  int pEndpointFromQdir;       // End Q vertex as viewed from beginning
                               // of Pdir relative to Pdir
  Vec firstIntersection;       // Point of intersection of Pdir, Qdir
  Vec secondIntersection;      // Second point of intersection
                               // (if there is one)
  int interiorFlag;            // Which polygon is inside the other
  int contained;               // Used for "completely contained" check
  int p_advances, q_advances;  // Number of times we've advanced
                               // P and Q indices
  int isFirstPoint;            // Is this the first point?
  int intersectionCode;        // SegSegIntersect() return code.

  // Check for Q contained in P

  contained = TRUE;

  for (ip = 0; ip < NP; ++ip) {
    ip_begin = (ip + NP - 1) % NP;

    Cross(&P[ip_begin], &P[ip], &Pdir);
    Normalize(&Pdir);

    for (iq = 0; iq < NQ; ++iq) {
      if (DEBUG >= 4) {
        printf("Q in P: Dot%d%d = %12.5e\n", ip, iq, Dot(&Pdir, &Q[iq]));
        fflush(stdout);
      }

      if (Dot(&Pdir, &Q[iq]) < -TOLERANCE) {
        contained = FALSE;
        break;
      }
    }

    if (!contained)
      break;
  }

  if (contained) {
    if (DEBUG >= 4) {
      printf("Q is entirely contained in P (output pixel is in input pixel)\n");
      fflush(stdout);
    }

    for (iq = 0; iq < NQ; ++iq)
      SaveVertex(&Q[iq], P, Q, V, nv);

    return;
  }

  // Check for P contained in Q

  contained = TRUE;

  for (iq = 0; iq < NQ; ++iq) {
    iq_begin = (iq + NQ - 1) % NQ;

    Cross(&Q[iq_begin], &Q[iq], &Qdir);
    Normalize(&Qdir);

    for (ip = 0; ip < NP; ++ip) {
      if (DEBUG >= 4) {
        printf("P in Q: Dot%d%d = %12.5e\n", iq, ip, Dot(&Qdir, &P[ip]));
        fflush(stdout);
      }

      if (Dot(&Qdir, &P[ip]) < -TOLERANCE) {
        contained = FALSE;
        break;
      }
    }

    if (!contained)
      break;
  }

  if (contained) {
    if (DEBUG >= 4) {
      printf("P is entirely contained in Q (input pixel is in output pixel)\n");
      fflush(stdout);
    }

    *nv = 0;
    for (ip = 0; ip < NP; ++ip)
      SaveVertex(&P[ip], P, Q, V, nv);

    return;
  }

  // Then check for polygon overlap

  ip = 0;
  iq = 0;

  p_advances = 0;
  q_advances = 0;

  interiorFlag = UNKNOWN;
  isFirstPoint = TRUE;

  while (FOREVER) {
    if (p_advances >= 2 * NP)
      break;
    if (q_advances >= 2 * NQ)
      break;
    if (p_advances >= NP && q_advances >= NQ)
      break;

    if (DEBUG >= 4) {
      printf("-----\n");

      if (interiorFlag == UNKNOWN) {
        printf("Before advances (UNKNOWN interiorFlag): ip=%d, iq=%d ", ip, iq);
        printf("(p_advances=%d, q_advances=%d)\n", p_advances, q_advances);
      }

      else if (interiorFlag == P_IN_Q) {
        printf("Before advances (P_IN_Q): ip=%d, iq=%d ", ip, iq);
        printf("(p_advances=%d, q_advances=%d)\n", p_advances, q_advances);
      }

      else if (interiorFlag == Q_IN_P) {
        printf("Before advances (Q_IN_P): ip=%d, iq=%d ", ip, iq);
        printf("(p_advances=%d, q_advances=%d)\n", p_advances, q_advances);
      } else
        printf("\nBAD INTERIOR FLAG.  Shouldn't get here\n");

      fflush(stdout);
    }

    // Previous point in the polygon

    ip_begin = (ip + NP - 1) % NP;
    iq_begin = (iq + NQ - 1) % NQ;

    // The current polygon edges are given by
    // the cross product of the vertex vectors

    Cross(&P[ip_begin], &P[ip], &Pdir);
    Cross(&Q[iq_begin], &Q[iq], &Qdir);

    PToQDir = DirectionCalculator(&P[ip], &Pdir, &Qdir);

    Cross(&Q[iq_begin], &P[ip], &other);
    pEndpointFromQdir = DirectionCalculator(&Q[iq_begin], &Qdir, &other);

    Cross(&P[ip_begin], &Q[iq], &other);
    qEndpointFromPdir = DirectionCalculator(&P[ip_begin], &Pdir, &other);

    if (DEBUG >= 4) {
      printf("   ");
      printDir("P", "Q", PToQDir);
      printDir("pEndpoint", "Q", pEndpointFromQdir);
      printDir("qEndpoint", "P", qEndpointFromPdir);
      printf("\n");
      fflush(stdout);
    }

    // Find point(s) of intersection between edges

    intersectionCode = SegSegIntersect(&Pdir, &Qdir, &P[ip_begin], &P[ip],
                                       &Q[iq_begin], &Q[iq], &firstIntersection,
                                       &secondIntersection);

    if (intersectionCode == NORMAL_INTERSECT
        || intersectionCode == ENDPOINT_ONLY) {
      if (interiorFlag == UNKNOWN && isFirstPoint) {
        p_advances = 0;
        q_advances = 0;

        isFirstPoint = FALSE;
      }

      interiorFlag = UpdateInteriorFlag(&firstIntersection, interiorFlag,
                                        pEndpointFromQdir, qEndpointFromPdir, P, Q, V, nv);

      if (DEBUG >= 4) {
        if (interiorFlag == UNKNOWN)
          printf("   interiorFlag -> UNKNOWN\n");

        else if (interiorFlag == P_IN_Q)
          printf("   interiorFlag -> P_IN_Q\n");

        else if (interiorFlag == Q_IN_P)
          printf("   interiorFlag -> Q_IN_P\n");

        else
          printf("   BAD interiorFlag.  Shouldn't get here\n");

        fflush(stdout);
      }
    }

    // Advance rules

    // Special case: Pdir & Qdir overlap and oppositely oriented.

    if ((intersectionCode == COLINEAR_SEGMENTS) && (Dot(&Pdir, &Qdir) < 0)) {
      if (DEBUG >= 4) {
        printf("   ADVANCE: Pdir and Qdir are colinear.\n");
        fflush(stdout);
      }

      SaveSharedSeg(&firstIntersection, &secondIntersection, P, Q, V, nv);

      RemoveDups(nv, V);
      return;
    }

    // Special case: Pdir & Qdir parallel and separated.

    if ((PToQDir == PARALLEL) && (pEndpointFromQdir == CLOCKWISE)
        && (qEndpointFromPdir == CLOCKWISE)) {
      if (DEBUG >= 4) {
        printf("   ADVANCE: Pdir and Qdir are disjoint.\n");
        fflush(stdout);
      }

      RemoveDups(nv, V);
      return;
    }

    // Special case: Pdir & Qdir colinear.

    else if ((PToQDir == PARALLEL) && (pEndpointFromQdir == PARALLEL)
        && (qEndpointFromPdir == PARALLEL)) {
      if (DEBUG >= 4) {
        printf("   ADVANCE: Pdir and Qdir are colinear.\n");
        fflush(stdout);
      }

      // Advance but do not output point.

      if (interiorFlag == P_IN_Q)
        iq = Advance(iq, &q_advances, NQ, interiorFlag == Q_IN_P, &Q[iq], P, Q, V, nv);
      else
        ip = Advance(ip, &p_advances, NP, interiorFlag == P_IN_Q, &P[ip], P, Q, V, nv);
    }

    // Generic cases.

    else if (PToQDir == COUNTERCLOCKWISE || PToQDir == PARALLEL) {
      if (qEndpointFromPdir == COUNTERCLOCKWISE) {
        if (DEBUG >= 4) {
          printf("   ADVANCE: Generic: PToQDir is COUNTERCLOCKWISE ");
          printf("|| PToQDir is PARALLEL, ");
          printf("qEndpointFromPdir is COUNTERCLOCKWISE\n");
          fflush(stdout);
        }

        ip = Advance(ip, &p_advances, NP, interiorFlag == P_IN_Q, &P[ip], P, Q, V, nv);
      } else {
        if (DEBUG >= 4) {
          printf("   ADVANCE: Generic: PToQDir is COUNTERCLOCKWISE ");
          printf("|| PToQDir is PARALLEL, qEndpointFromPdir is CLOCKWISE\n");
          fflush(stdout);
        }

        iq = Advance(iq, &q_advances, NQ, interiorFlag == Q_IN_P, &Q[iq], P, Q, V, nv);
      }
    }

    else {
      if (pEndpointFromQdir == COUNTERCLOCKWISE) {
        if (DEBUG >= 4) {
          printf("   ADVANCE: Generic: PToQDir is CLOCKWISE, ");
          printf("pEndpointFromQdir is COUNTERCLOCKWISE\n");
          fflush(stdout);
        }

        iq = Advance(iq, &q_advances, NQ, interiorFlag == Q_IN_P, &Q[iq], P, Q, V, nv);
      } else {
        if (DEBUG >= 4) {
          printf("   ADVANCE: Generic: PToQDir is CLOCKWISE, ");
          printf("pEndpointFromQdir is CLOCKWISE\n");
          fflush(stdout);
        }

        ip = Advance(ip, &p_advances, NP, interiorFlag == P_IN_Q, &P[ip], P, Q, V, nv);
      }
    }

    if (DEBUG >= 4) {
      if (interiorFlag == UNKNOWN) {
        printf("After  advances: ip=%d, iq=%d ", ip, iq);
        printf("(p_advances=%d, q_advances=%d) interiorFlag=UNKNOWN\n",
               p_advances, q_advances);
      }

      else if (interiorFlag == P_IN_Q) {
        printf("After  advances: ip=%d, iq=%d ", ip, iq);
        printf("(p_advances=%d, q_advances=%d) interiorFlag=P_IN_Q\n",
               p_advances, q_advances);
      }

      else if (interiorFlag == Q_IN_P) {
        printf("After  advances: ip=%d, iq=%d ", ip, iq);
        printf("(p_advances=%d, q_advances=%d) interiorFlag=Q_IN_P\n",
               p_advances, q_advances);
      } else
        printf("BAD INTERIOR FLAG.  Shouldn't get here\n");

      printf("-----\n\n");
      fflush(stdout);
    }
  }

  RemoveDups(nv, V);
  return;
}

/*
 * Print out the second point of intersection and toggle in/out flag.
 */
int UpdateInteriorFlag(Vec *p, int interiorFlag, int pEndpointFromQdir,
                       int qEndpointFromPdir, Vec *P, Vec *Q, Vec *V, int *nv) {
  double lon, lat;

  if (DEBUG >= 4) {
    lon = atan2(p->y, p->x) / DEG_TO_RADIANS;
    lat = asin(p->z) / DEG_TO_RADIANS;

    printf("   intersection [%13.6e,%13.6e,%13.6e]  "
           "-> (%10.6f,%10.6f) (UpdateInteriorFlag)\n",
           p->x, p->y, p->z, lon, lat);
    fflush(stdout);
  }

  SaveVertex(p, P, Q, V, nv);

  // Update interiorFlag.

  if (pEndpointFromQdir == COUNTERCLOCKWISE)
    return P_IN_Q;

  else if (qEndpointFromPdir == COUNTERCLOCKWISE)
    return Q_IN_P;

  else
    // Keep status quo.
    return interiorFlag;
}

/*
 * Save the endpoints of a shared segment.
 */
void SaveSharedSeg(Vec *p, Vec *q, Vec *P, Vec *Q, Vec *V, int *nv) {
  if (DEBUG >= 4) {
    printf("\n   SaveSharedSeg():  from "
           "[%13.6e,%13.6e,%13.6e]\n",
           p->x, p->y, p->z);

    printf("   SaveSharedSeg():  to   "
           "[%13.6e,%13.6e,%13.6e]\n\n",
           q->x, q->y, q->z);

    fflush(stdout);
  }

  SaveVertex(p, P, Q, V, nv);
  SaveVertex(q, P, Q, V, nv);

}

/*
 * Advances and prints out an inside vertex if appropriate.
 */
int Advance(int ip, int *p_advances, int n, int inside, Vec *v, Vec *P, Vec *Q, Vec *V, int *nv) {
  double lon, lat;

  lon = atan2(v->y, v->x) / DEG_TO_RADIANS;
  lat = asin(v->z) / DEG_TO_RADIANS;

  if (inside) {
    if (DEBUG >= 4) {
      printf("   Advance(): inside vertex "
             "[%13.6e,%13.6e,%13.6e] -> (%10.6f,%10.6f)n",
             v->x, v->y, v->z, lon, lat);

      fflush(stdout);
    }

    SaveVertex(v, P, Q, V, nv);
  }

  (*p_advances)++;

  return (ip + 1) % n;
}

/*
 * Save the intersection polygon vertices
 */
void SaveVertex(Vec *a, Vec *P, Vec *Q, Vec *V, int *nv) {
  int i, i_begin;
  Vec Dir;

  if (DEBUG >= 4)
    printf("   SaveVertex ... ");

  // What with TOLERANCE and roundoff problems, we need to double-check
  // that the point to be save is really in or on the edge of both pixels P and Q.

  for (i = 0; i < NP; ++i) {
    i_begin = (i + NP - 1) % NP;

    Cross(&P[i_begin], &P[i], &Dir);
    Normalize(&Dir);

    if (Dot(&Dir, a) < -1000. * TOLERANCE) {
      if (DEBUG >= 4) {
        printf("rejected (not in P)\n");
        fflush(stdout);
      }

      return;
    }
  }

  for (i = 0; i < NQ; ++i) {
    i_begin = (i + NQ - 1) % NQ;

    Cross(&Q[i_begin], &Q[i], &Dir);
    Normalize(&Dir);

    if (Dot(&Dir, a) < -1000. * TOLERANCE) {
      if (DEBUG >= 4) {
        printf("rejected (not in Q)\n");
        fflush(stdout);
      }

      return;
    }
  }

  if (*nv < 15) {
    V[*nv].x = a->x;
    V[*nv].y = a->y;
    V[*nv].z = a->z;
    *nv += 1;
  }

  if (DEBUG >= 4) {
    printf("accepted (%d)\n", *nv);
    fflush(stdout);
  }
}

/*
 * Computes whether ac is CLOCKWISE, etc. of ab.
 */
int DirectionCalculator(Vec *a, Vec *b, Vec *c) {
  Vec cross;
  int len;

  len = Cross(b, c, &cross);

  if (len == 0)
    return PARALLEL;
  else if (Dot(a, &cross) < 0.)
    return CLOCKWISE;
  else
    return COUNTERCLOCKWISE;
}

/*
 * Finds the point of intersection p between two closed segments ab and cd.
 *
 * Returns p and a char with the following meaning:
 *
 *   COLINEAR_SEGMENTS: The segments colinearly overlap, sharing a point.
 *
 *   ENDPOINT_ONLY:     An endpoint (vertex) of one segment is on the other
 *                      segment, but COLINEAR_SEGMENTS doesn't hold.
 *
 *   NORMAL_INTERSECT:  The segments intersect properly (i.e., they share
 *                      a point and neither ENDPOINT_ONLY nor
 *                      COLINEAR_SEGMENTS holds).
 *
 *   NO_INTERSECTION:   The segments do not intersect (i.e., they share
 *                      no points).
 *
 * Note that two colinear segments that share just one point, an endpoint
 * of each, returns COLINEAR_SEGMENTS rather than ENDPOINT_ONLY as one
 * might expect.
 */
int SegSegIntersect(Vec *pEdge, Vec *qEdge, Vec *p0, Vec *p1, Vec *q0, Vec *q1,
                    Vec *intersect1, Vec *intersect2) {
  double pDot, qDot;   // Dot product [cos(length)] of the edge vertices
  double p0Dot, p1Dot;  // Dot product from vertices to intersection
  double q0Dot, q1Dot;  // Dot pro}duct from vertices to intersection
  int len;

  // Get the edge lengths (actually cos(length))

  pDot = Dot(p0, p1);
  qDot = Dot(q0, q1);

  // Find the point of intersection

  len = Cross(pEdge, qEdge, intersect1);

  // If the two edges are colinear, check to see if they overlap

  if (len == 0) {
    if (Between(q0, p0, p1) && Between(q1, p0, p1)) {
      intersect1 = q0;
      intersect2 = q1;
      return COLINEAR_SEGMENTS;
    }

    if (Between(p0, q0, q1) && Between(p1, q0, q1)) {
      intersect1 = p0;
      intersect2 = p1;
      return COLINEAR_SEGMENTS;
    }

    if (Between(q0, p0, p1) && Between(p1, q0, q1)) {
      intersect1 = q0;
      intersect2 = p1;
      return COLINEAR_SEGMENTS;
    }

    if (Between(p0, q0, q1) && Between(q1, p0, p1)) {
      intersect1 = p0;
      intersect2 = q1;
      return COLINEAR_SEGMENTS;
    }

    if (Between(q1, p0, p1) && Between(p1, q0, q1)) {
      intersect1 = p0;
      intersect2 = p1;
      return COLINEAR_SEGMENTS;
    }

    if (Between(q0, p0, p1) && Between(p0, q0, q1)) {
      intersect1 = p0;
      intersect2 = q0;
      return COLINEAR_SEGMENTS;
    }

    return NO_INTERSECTION;
  }

  // If this is the wrong one of the two
  // (other side of the sky) reverse it

  Normalize(intersect1);

  if (Dot(intersect1, p0) < 0.)
    Reverse(intersect1);

  // Point has to be inside both sides to be an intersection

  if ((p0Dot = Dot(intersect1, p0)) < pDot)
    return NO_INTERSECTION;
  if ((p1Dot = Dot(intersect1, p1)) < pDot)
    return NO_INTERSECTION;
  if ((q0Dot = Dot(intersect1, q0)) < qDot)
    return NO_INTERSECTION;
  if ((q1Dot = Dot(intersect1, q1)) < qDot)
    return NO_INTERSECTION;

  // Otherwise, if the intersection is at an endpoint

  if (p0Dot == pDot)
    return ENDPOINT_ONLY;
  if (p1Dot == pDot)
    return ENDPOINT_ONLY;
  if (q0Dot == qDot)
    return ENDPOINT_ONLY;
  if (q1Dot == qDot)
    return ENDPOINT_ONLY;

  // Otherwise, it is a normal intersection

  return NORMAL_INTERSECT;
}

/*
 * Formats a message about relative directions.
 */
int printDir(char *point, char *vector, int dir) {
  if (dir == CLOCKWISE)
    printf("%s is CLOCKWISE of %s; ", point, vector);

  else if (dir == COUNTERCLOCKWISE)
    printf("%s is COUNTERCLOCKWISE of %s; ", point, vector);

  else if (dir == PARALLEL)
    printf("%s is PARALLEL to %s; ", point, vector);

  else
    printf("Bad comparison (shouldn't get this; ");

  return 0;
}

/*
 * Tests whether whether a point on an arc is
 * between two other points.
 */
int Between(Vec *v, Vec *a, Vec *b) {
  double abDot, avDot, bvDot;

  abDot = Dot(a, b);
  avDot = Dot(a, v);
  bvDot = Dot(b, v);

  if (avDot > abDot && bvDot > abDot)
    return TRUE;
  else
    return FALSE;
}

/*
 * Vector cross product.
 */
int Cross(Vec *v1, Vec *v2, Vec *v3) {
  v3->x = v1->y * v2->z - v2->y * v1->z;
  v3->y = -v1->x * v2->z + v2->x * v1->z;
  v3->z = v1->x * v2->y - v2->x * v1->y;

  if (fabs(v3->x) < 1.e-18 && fabs(v3->y) < 1.e-18 && fabs(v3->z) < 1.e-18)
    return 0;

  return 1;
}

/*
 * Vector dot product.
 */
double Dot(Vec *a, Vec *b) {
  double sum = 0.0;

  sum = a->x * b->x + a->y * b->y + a->z * b->z;

  return sum;
}

/*
 * Normalize the vector
 */
double Normalize(Vec *v) {
  double len;

  len = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);

  if (len == 0.)
    len = 1.;

  v->x = v->x / len;
  v->y = v->y / len;
  v->z = v->z / len;

  return len;
}

/*
 * Reverse the vector.
 */
void Reverse(Vec *v) {
  v->x = -v->x;
  v->y = -v->y;
  v->z = -v->z;
}

/*
 * Use Girard's theorem to compute the area of a sky polygon.
 */
double Girard(int nv, Vec *V) {
  int i, j, ibad;

  double area;
  double lon, lat;

  Vec side[16];
  double ang[16];

  Vec tmp;

  double sumang, cosAng, sinAng;

  sumang = 0;

  if (nv < 3)
    return 0;

  if (DEBUG >= 4) {
    for (i = 0; i < nv; ++i) {
      lon = atan2(V[i].y, V[i].x) / DEG_TO_RADIANS;
      lat = asin(V[i].z) / DEG_TO_RADIANS;

      printf("Girard(): %3d [%13.6e,%13.6e,%13.6e] -> (%10.6f,%10.6f)\n", i,
             V[i].x, V[i].y, V[i].z, lon, lat);

      fflush(stdout);
    }
  }

  for (i = 0; i < nv; ++i) {
    Cross(&V[i], &V[(i + 1) % nv], &side[i]);
  }

  // De-duplicate vertices that are extremely close to each other otherwise
  // the angles determined in the next steps are not accurate. Need to loop
  // backwards to avoid affecting future sides that need to be checked.

  for (i = nv - 1; i >= 0; --i) {

    // We don't use TOLERANCE here since it is too large for our purposes here

    if (Dot(&side[i], &side[i]) < 1e-30) {

      if (DEBUG >= 4) {
        printf("Girard(): ---------- Corner %d duplicate; "
               "Remove point %d -------------\n",
               i, i);
        fflush(stdout);
      }

      --nv;

      for (j = i; j < nv; ++j) {
        V[j].x = V[j + 1].x;
        V[j].y = V[j + 1].y;
        V[j].z = V[j + 1].z;
        side[j].x = side[j + 1].x;
        side[j].y = side[j + 1].y;
        side[j].z = side[j + 1].z;
      }

    }
  }

  if (nv < 3)
    return 0;

  for (i = 0; i < nv; ++i) {
    Normalize(&side[i]);
  }

  for (i = 0; i < nv; ++i) {
    Cross(&side[i], &side[(i + 1) % nv], &tmp);

    sinAng = Normalize(&tmp);
    cosAng = -Dot(&side[i], &side[(i + 1) % nv]);

    // Remove center point of colinear segments

    ang[i] = atan2(sinAng, cosAng);

    if (DEBUG >= 4) {
      if (i == 0)
        printf("\n");

      printf("Girard(): angle[%d] = %13.6e -> %13.6e (from %13.6e / %13.6e)\n",
             i, ang[i], ang[i] - M_PI / 2., sinAng, cosAng);
      fflush(stdout);
    }

    // Direction changes of less than a degree can be tricky
    if (ang[i] > M_PI - 0.0175) {
      ibad = (i + 1) % nv;

      if (DEBUG >= 4) {
        printf("Girard(): ---------- Corner %d bad; "
               "Remove point %d -------------\n",
               i, ibad);
        fflush(stdout);
      }

      --nv;

      for (j = ibad; j < nv; ++j) {
        V[j].x = V[j + 1].x;
        V[j].y = V[j + 1].y;
        V[j].z = V[j + 1].z;
      }

      return (Girard(nv, V));
    }

    sumang += ang[i];
  }

  area = sumang - (nv - 2.) * M_PI;

  if (mNaN(area) || area < 0.)
    area = 0.;

  if (DEBUG >= 4) {
    printf("\nGirard(): area = %13.6e [%d]\n\n", area, nv);
    fflush(stdout);
  }

  return area;
}

/*
 * Check the vertex list for adjacent pairs of
 * points which are too close together for the
 * subsequent dot- and cross-product calculations
 * of Girard's theorem.
 */
void RemoveDups(int *nv, Vec *V) {
  int i, nvnew;
  Vec Vnew[16];
  Vec tmp;
  double lon, lat;

  double separation;

  if (DEBUG >= 4) {
    printf("RemoveDups() TOLERANCE = %13.6e [%13.6e arcsec]\n\n", TOLERANCE,
           TOLERANCE / DEG_TO_RADIANS * 3600.);

    for (i = 0; i < *nv; ++i) {
      lon = atan2(V[i].y, V[i].x) / DEG_TO_RADIANS;
      lat = asin(V[i].z) / DEG_TO_RADIANS;

      printf("RemoveDups() orig: %3d [%13.6e,%13.6e,%13.6e] "
             "-> (%10.6f,%10.6f)\n",
             i, V[i].x, V[i].y, V[i].z, lon, lat);

      fflush(stdout);
    }

    printf("\n");
  }

  Vnew[0].x = V[0].x;
  Vnew[0].y = V[0].y;
  Vnew[0].z = V[0].z;

  nvnew = 0;

  for (i = 0; i < *nv; ++i) {
    ++nvnew;

    Vnew[nvnew].x = V[(i + 1) % *nv].x;
    Vnew[nvnew].y = V[(i + 1) % *nv].y;
    Vnew[nvnew].z = V[(i + 1) % *nv].z;

    Cross(&V[i], &V[(i + 1) % *nv], &tmp);

    separation = Normalize(&tmp);

    if (DEBUG >= 4) {
      printf("RemoveDups(): %3d x %3d: distance = %13.6e "
             "[%13.6e arcsec] (would become %d)\n",
             (i + 1) % *nv, i, separation, separation / DEG_TO_RADIANS * 3600.,
             nvnew);

      fflush(stdout);
    }

    if (separation < TOLERANCE) {
      --nvnew;

      if (DEBUG >= 4) {
        printf("RemoveDups(): %3d is a duplicate (nvnew -> %d)\n", i, nvnew);

        fflush(stdout);
      }
    }
  }

  if (DEBUG >= 4) {
    printf("\n");
    fflush(stdout);
  }

  if (nvnew < *nv) {
    for (i = 0; i < nvnew; ++i) {
      V[i].x = Vnew[i].x;
      V[i].y = Vnew[i].y;
      V[i].z = Vnew[i].z;
    }

    *nv = nvnew;
  }
}
