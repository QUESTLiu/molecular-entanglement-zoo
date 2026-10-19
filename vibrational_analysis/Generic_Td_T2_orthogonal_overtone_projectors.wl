(* ::Package:: *)

(* ============================================================ *)
(*  Td Overtone Projectors for Methane T2 Vibrational Irrep      *)
(*                                                              *)
(*  Case: Sym^N(T2)                                  *)
(*                                                              *)
(*  This file constructs projectors for the symmetric overtone   *)
(*  manifold of a triply degenerate T2 normal-mode block in Td.  *)
(*                                                              *)
(*      Sym^2(T2) = A1 ⊕ E ⊕ T2                                 *)
(*      Sym^3(T2) = A1 ⊕ T1 ⊕ 2 T2                              *)
(*                                                              *)
(*  The T2 representation is the standard orthonormal coordinate *)
(*  representation of Td acting on (x,y,z). It is generated as   *)
(*  signed permutation matrices preserving the tetrahedron.      *)
(* ============================================================ *)

ClearAll["Global`*"];

(* Change degree to 2, 3, or higher as needed. *)
degree = 2;

Print["============================================================"];
Print["Target overtone manifold: Sym^", degree, "(T2) in Td"];
Print["============================================================"];

(* ============================================================ *)
(* 1. Generate the 24 operations of Td as orthogonal matrices    *)
(* ============================================================ *)

(* Td can be represented by the 24 signed permutation matrices
   with product of signs equal to +1. These preserve the vertex set
   of a regular tetrahedron:
      (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
   This 3D representation has characters:
      {E, 8C3, 3C2, 6S4, 6sigma_d} = {3,0,-1,-1,1},
   which is the T2 irrep of Td.
*)

permutationMatrix[p_] := Module[{n = Length[p]},
  Normal[SparseArray[Table[{i, p[[i]]} -> 1, {i, n}], {n, n}]]
];

signedPermutationMatrices3D[] := Module[{perms, signs, mats},
  perms = Permutations[Range[3]];
  signs = Select[Tuples[{-1, 1}, 3], Times @@ # == 1 &];

  mats = Flatten[
    Table[
      DiagonalMatrix[s].permutationMatrix[p],
      {p, perms}, {s, signs}
    ],
    1
  ];

  DeleteDuplicates[mats]
];

tdElements = signedPermutationMatrices3D[];

Print["Number of Td elements = ", Length[tdElements]];

If[Length[tdElements] != 24,
  Print["WARNING: Expected 24 Td operations."]
];

Print["All T2 matrices are orthogonal: ",
  And @@ (FullSimplify[Transpose[#].# == IdentityMatrix[3]] & /@ tdElements)
];

(* ============================================================ *)
(* 2. Class labels and character table for Td                    *)
(* ============================================================ *)

classLabel[g_] := Module[{det = Det[g], tr = Tr[g]},
  Which[
    det == 1 && tr == 3, "E",
    det == 1 && tr == 0, "C3",
    det == 1 && tr == -1, "C2",
    det == -1 && tr == -1, "S4",
    det == -1 && tr == 1, "sigmad",
    True, "Unknown"
  ]
];

Print["Class tally:"];
Print[Tally[classLabel /@ tdElements]];

irrepNames = {"A1", "A2", "E", "T1", "T2"};

irrepDim = <|
  "A1" -> 1,
  "A2" -> 1,
  "E" -> 2,
  "T1" -> 3,
  "T2" -> 3
|>;

char[ir_, g_] := Module[{c = classLabel[g]},
  Switch[ir,
    "A1",
      Switch[c, "E", 1, "C3", 1, "C2", 1, "S4", 1, "sigmad", 1],
    "A2",
      Switch[c, "E", 1, "C3", 1, "C2", 1, "S4", -1, "sigmad", -1],
    "E",
      Switch[c, "E", 2, "C3", -1, "C2", 2, "S4", 0, "sigmad", 0],
    "T1",
      Switch[c, "E", 3, "C3", 0, "C2", -1, "S4", 1, "sigmad", -1],
    "T2",
      Switch[c, "E", 3, "C3", 0, "C2", -1, "S4", -1, "sigmad", 1]
  ]
];

Print["T2 character tally from matrices:"];
Print[Sort[Tally[Tr /@ tdElements]]];

(* ============================================================ *)
(* 3. Symmetric-power/Fock representation builder                *)
(* ============================================================ *)

exponentTuples[d_, total_] := SortBy[
  Select[Tuples[Range[0, total], d], Total[#] == total &],
  -# &
];

factorialProduct[nvec_] := Times @@ (Factorial /@ nvec);

rawMonomial[vars_, nvec_] := Times @@ MapThread[#1^#2 &, {vars, nvec}];

fockLabel[nvec_] := StringJoin["|", StringJoin[ToString /@ nvec], ">"];

buildSymPowerRep[matrixList_, total_] := Module[
  {d, vars, exps, labels, norms, rawMonomials, basis, extractCoeffs, getRep},

  d = Length[matrixList[[1]]];
  vars = Array[q, d];
  exps = exponentTuples[d, total];
  labels = fockLabel /@ exps;

  norms = 1/Sqrt[factorialProduct[#]] & /@ exps;
  rawMonomials = rawMonomial[vars, #] & /@ exps;
  basis = norms*rawMonomials;

  extractCoeffs[expr_] := Module[{rules, dict},
    rules = CoefficientRules[Expand[expr], vars];
    dict = Association[rules];
    Table[Lookup[dict, Key[exp], 0], {exp, exps}]
  ];

  getRep[g_] := Module[{transVars, transBasis},
    (* Orthonormal coordinate/Fock convention:
       q_i -> Sum_j g[[i,j]] q_j.
    *)
    transVars = g.vars;
    transBasis = Expand[basis /. Thread[vars -> transVars]];

    Table[
      extractCoeffs[transBasis[[i]]][[j]]/norms[[j]],
      {i, 1, Length[exps]},
      {j, 1, Length[exps]}
    ]
  ];

  <|
    "Dimension" -> Binomial[d + total - 1, total],
    "Exponents" -> exps,
    "Labels" -> labels,
    "Matrices" -> FullSimplify[getRep /@ matrixList]
  |>
];

(* ============================================================ *)
(* 4. Build Sym^degree(T2) representation                        *)
(* ============================================================ *)

Print["\nBuilding Sym^", degree, "(T2) representation..."];

symData = buildSymPowerRep[tdElements, degree];

fockExponents = symData["Exponents"];
fockLabels = symData["Labels"];
repN = symData["Matrices"];
dimN = symData["Dimension"];

Print["Fock-space dimension = ", dimN];

Print["\nFock basis order:"];
Print[Column[Thread[Range[Length[fockLabels]] -> fockLabels]]];

Print["Induced matrices are orthogonal: ",
  And @@ (FullSimplify[Transpose[#].# == IdentityMatrix[dimN]] & /@ repN)
];

(* ============================================================ *)
(* 5. Character decomposition                                    *)
(* ============================================================ *)

chiRep[k_] := FullSimplify[Tr[repN[[k]]]];

multiplicity[ir_] := FullSimplify[
  (1/Length[tdElements]) Sum[
    char[ir, tdElements[[k]]] chiRep[k],
    {k, 1, Length[tdElements]}
  ]
];

multiplicities = Association[
  Table[ir -> multiplicity[ir], {ir, irrepNames}]
];

activeIrreps = Select[irrepNames, Not[PossibleZeroQ[multiplicities[#]]] &];

Print["\n--- Character Decomposition ---"];
Print[
  Row[
    Table[
      If[multiplicities[ir] === 0,
        Nothing,
        ToString[TraditionalForm[multiplicities[ir]]] <> " " <> ir <> "  "
      ],
      {ir, irrepNames}
    ],
    " + "
  ]
];

Print["Multiplicities:"];
Print[multiplicities];

If[Max[Values[multiplicities]] > 1,
  Print["\nNOTE: At least one irrep appears with multiplicity > 1."];
  Print["The character projector returns the full isotypic subspace, not a unique individual copy."];
];

(* ============================================================ *)
(* 6. Construct projectors                                       *)
(* ============================================================ *)

projector[ir_] := FullSimplify[
  (irrepDim[ir]/Length[tdElements]) Sum[
    char[ir, tdElements[[k]]] repN[[k]],
    {k, 1, Length[tdElements]}
  ]
];

projectors = Association[
  Table[ir -> projector[ir], {ir, activeIrreps}]
];

Print["\n--- Projector Checks ---"];

Do[
  Print["P_", ir, "^2 = P_", ir, ": ",
    FullSimplify[projectors[ir].projectors[ir] == projectors[ir]]
  ];
  Print["P_", ir, " symmetric: ",
    FullSimplify[Transpose[projectors[ir]] == projectors[ir]]
  ];
  Print["Rank P_", ir, " = ",
    MatrixRank[projectors[ir]],
    "  expected = ",
    multiplicities[ir] irrepDim[ir]
  ];
  ,
  {ir, activeIrreps}
];

Print["Completeness of active projectors: ",
  FullSimplify[
    Total[Values[projectors]] == IdentityMatrix[dimN]
  ]
];

Print["Mutual orthogonality of active projectors:"];

Do[
  If[i < j,
    Print["P_", activeIrreps[[i]],
      " P_", activeIrreps[[j]], " = 0: ",
      FullSimplify[
        projectors[activeIrreps[[i]]].projectors[activeIrreps[[j]]] ==
        ConstantArray[0, {dimN, dimN}]
      ]
    ]
  ],
  {i, Length[activeIrreps]},
  {j, Length[activeIrreps]}
];

(* ============================================================ *)
(* 7. Extract orthonormal symmetry-adapted basis states          *)
(* ============================================================ *)

extractBasis[proj_] := Module[{rows, nonzeroRows},
  rows = RowReduce[FullSimplify[proj]];
  nonzeroRows = Select[rows, Not[PossibleZeroQ[FullSimplify[#.#]]] &];
  FullSimplify[Orthogonalize[nonzeroRows]]
];

basisByIrrep = Association[
  Table[ir -> extractBasis[projectors[ir]], {ir, activeIrreps}]
];

Print["\n--- Symmetry-Adapted Basis Dimensions ---"];

Do[
  Print[ir, " basis dimension = ", Length[basisByIrrep[ir]]],
  {ir, activeIrreps}
];

ClearAll[Ket];

stateExpression[v_] := FullSimplify[
  Total[MapThread[#1 Ket[#2] &, {v, fockLabels}]]
];

statesByIrrep = Association[
  Table[ir -> (stateExpression /@ basisByIrrep[ir]), {ir, activeIrreps}]
];

Print["\n--- Symmetry-Adapted Fock States ---"];

Do[
  Print["\n", ir, " states:"];
  Print[Column[statesByIrrep[ir]]];
  ,
  {ir, activeIrreps}
];

(* ============================================================ *)
(* 8. Optional block diagonalization check                       *)
(* ============================================================ *)

U = Join @@ (basisByIrrep /@ activeIrreps);

Print["\nBlock-diagonalizing matrix U dimensions: ", Dimensions[U]];
Print["U is orthogonal: ", FullSimplify[U.Transpose[U] == IdentityMatrix[dimN]]];

rhoBlock[k_] := FullSimplify[U.repN[[k]].Transpose[U]];

sampleIndex = 2;

Print["\nSample block-diagonalized representation matrix:"];
Print[MatrixForm[rhoBlock[sampleIndex]]];

Print["\nDone."];
