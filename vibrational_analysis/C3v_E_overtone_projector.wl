(* ::Package:: *)

ClearAll["Global`*"];

omega = Exp[2 Pi I/3];

(* Change this number only *)
userOvertone = 6;

groupElements = {"E", "C3", "C3sq", "sv1", "sv2", "sv3"};

chars = <|
   "A1" -> {1, 1, 1, 1, 1, 1},
   "A2" -> {1, 1, 1, -1, -1, -1},
   "E"  -> {2, -1, -1, 0, 0, 0}
|>;

irrepDim = <|"A1" -> 1, "A2" -> 1, "E" -> 2|>;
groupOrder = 6;

fixedBasis[n_] := Table[{np, n - np}, {np, 0, n}];

act[g_, {np_, nm_}] := Switch[g,
   "E", {1, {np, nm}},
   "C3", {omega^(np - nm), {np, nm}},
   "C3sq", {omega^(2 (np - nm)), {np, nm}},
   "sv1", {1, {nm, np}},
   "sv2", {omega^(np - nm), {nm, np}},
   "sv3", {omega^(2 (np - nm)), {nm, np}}
];

repMatrix[g_, basis_] := Module[{M, coeff, image, pos},
   M = ConstantArray[0, {Length[basis], Length[basis]}];
   Do[
    {coeff, image} = act[g, basis[[j]]];
    pos = FirstPosition[basis, image][[1]];
    M[[pos, j]] = coeff;
    ,
    {j, Length[basis]}
   ];
   Simplify[M]
];

projector[irrep_, basis_] := FullSimplify[
   irrepDim[irrep]/groupOrder Sum[
     chars[irrep][[k]] repMatrix[groupElements[[k]], basis],
     {k, Length[groupElements]}
   ]
];

adaptedBasis[irrep_, n_] := Module[
   {basis, P, projected, nonzero, ortho},
   basis = fixedBasis[n];
   P = projector[irrep, basis];
   projected = P . # & /@ IdentityMatrix[Length[basis]];
   nonzero = Select[projected, Norm[N[#]] > 10^-10 &];
   ortho = Orthogonalize[nonzero];
   Select[ortho, Norm[N[#]] > 10^-10 &]
];

stateString[vec_, basis_] := Module[{terms},
   terms = Table[
     If[Chop[N[vec[[i]]]] == 0,
      Nothing,
      ToString[InputForm[FullSimplify[vec[[i]]]]] <>
       " |" <> ToString[basis[[i, 1]]] <> "," <>
       ToString[basis[[i, 2]]] <> ">"
     ],
     {i, Length[basis]}
   ];
   If[terms == {}, "0", StringRiffle[terms, " + "]]
];

basis = fixedBasis[userOvertone];

A1basis = adaptedBasis["A1", userOvertone];
A2basis = adaptedBasis["A2", userOvertone];
Ebasis = adaptedBasis["E", userOvertone];

Print["========================================"];
Print["User-defined overtone order n = ", userOvertone];
Print["Original basis {nPlus,nMinus}: ", basis];

Print["A1 adapted basis:"];
Print[stateString[#, basis] & /@ A1basis];

Print["A2 adapted basis:"];
Print[stateString[#, basis] & /@ A2basis];

Print["E adapted basis:"];
Print[stateString[#, basis] & /@ Ebasis];

Print["Done."];



