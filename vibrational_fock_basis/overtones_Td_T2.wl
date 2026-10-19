(* ::Package:: *)

(* ============================================================================ *)
(*  overtones_Td_T2.wl                                                          *)
(*                                                                              *)
(*  Symmetry-adapted states of the overtone manifold Sym^N(T2) of ONE triply    *)
(*  degenerate T2 mode in Td (e.g. nu3 or nu4 of CH4).                           *)
(*                                                                              *)
(*      Sym^2(T2) = A1 + E + T2,   Sym^3(T2) = A1 + T1 + 2 T2,  ...             *)
(*                                                                              *)
(*  Conventions (main text):                                                     *)
(*   * Td = the 24 signed permutation matrices R of (x,y,z) with sign product +1 *)
(*     (the vertices (1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1) are preserved).      *)
(*   * Fock action, Lemma 2 / Eq. (21):  U(R) a_i^dag U(R)^dag = Sum_j R_ji a_j^dag,*)
(*     so a single quantum |1_i> = a_i^dag|0> is the Cartesian T2 partner i.    *)
(*   * Irrep matrices in the basis used for rotation and nuclear spin:          *)
(*        A1 = 1,  A2 = det R,  T2 = R (x,y,z),  T1 = det R * R,                 *)
(*        E  in the complex basis |1E>, |2E> with C3[111] (x->y->z) = diag(w,w^2) *)
(*        and the x<->y reflection a pure swap; on two quanta                   *)
(*        |1,2 E> = 3^(-1/2) Sum_i w^(-+i) |2_i>   (i = x,y,z = 1,2,3).          *)
(*   * Symmetry-adapted states from the matrix-valued projectors                *)
(*        P^G_{mu nu} = d_G/|G| Sum_R conj(D^G_{mu nu}(R)) U(R)   (Theorem 1, Eq. (25)),*)
(*     copies k = 1..m_G from the range of P^G_{11}, partners |mu G;k> = P^G_{mu 1}|1 G;k>.*)
(*     Repeated irreps (e.g. 2 T2 in Sym^3) are thus split into individual,     *)
(*     partner-resolved copies; the copy basis is fixed only up to a U(m_G)     *)
(*     rotation (Theorem 1 (iii)).                                              *)
(*                                                                              *)
(*  Run:  wolframscript -file overtones_Td_T2.wl                                *)
(* ============================================================================ *)

ClearAll["Global`*"];

(* ---------------- user input: number of quanta N ---------------- *)
degree = 2;
(* ----------------------------------------------------------------- *)

w = Exp[2 Pi I/3];

(* ---------- 1. Td as signed permutation matrices ---------- *)
permMat[p_] := Normal[SparseArray[Table[{p[[i]], i} -> 1, {i, 3}], {3, 3}]];   (* e_i -> e_p(i) *)
td = DeleteDuplicates@Flatten[Table[DiagonalMatrix[s] . permMat[p],
     {p, Permutations[Range[3]]}, {s, Select[Tuples[{-1, 1}, 3], Times @@ # == 1 &]}], 1];
nG = Length[td];
classOf[g_] := Which[Det[g] == 1 && Tr[g] == 3, "E", Det[g] == 1 && Tr[g] == 0, "8C3",
   Det[g] == 1 && Tr[g] == -1, "3C2", Det[g] == -1 && Tr[g] == -1, "6S4", Det[g] == -1 && Tr[g] == 1, "6sd"];
Print["|Td| = ", nG, ",  classes ", Tally[classOf /@ td]];

(* ---------- 2. Fock representation on N quanta (Eq. (21)/(22)) ---------- *)
focks[nq_] := ReverseSort[FrobeniusSolve[{1, 1, 1}, nq]];
fockMatrix[R_, nq_] := Module[{x, vars = Array[x, 3], basis = focks[nq], img, rules},
   Transpose@Table[
     img = Expand[Times @@ ((Transpose[R] . vars)^m)/Sqrt[Times @@ (m!)]];
     rules = Association[CoefficientRules[img, vars]];
     Table[Lookup[rules, Key[k], 0] Sqrt[Times @@ (k!)], {k, basis}],
     {m, basis}]];

basisN = focks[degree];
dimN = Length[basisN];
UN = fockMatrix[#, degree] & /@ td;
idx = AssociationThread[td -> Range[nG]];
Print["U is a representation (U(R)U(S) = U(RS) for all R,S): ",
  And @@ Flatten@Table[UN[[i]] . UN[[j]] == UN[[idx[td[[i]] . td[[j]]]]], {i, nG}, {j, nG}]];

(* ---------- 3. irrep matrices in the paper's basis ---------- *)
(* E: action on |1E>,|2E> inside the two-quantum manifold *)
basis2 = focks[2];
U2 = fockMatrix[#, 2] & /@ td;
vec2[coeffs_] := Table[Which[k == {2, 0, 0}, coeffs[[1]], k == {0, 2, 0}, coeffs[[2]], k == {0, 0, 2}, coeffs[[3]],
     True, 0], {k, basis2}];
e1 = vec2[Table[w^(-i), {i, 3}]]/Sqrt[3];
e2 = vec2[Table[w^(i), {i, 3}]]/Sqrt[3];
EB = {e1, e2};
DE = Table[Simplify[Table[Conjugate[EB[[mu]]] . U2[[g]] . EB[[nu]], {mu, 2}, {nu, 2}]], {g, nG}];
irrepMats = <|
   "A1" -> Table[{{1}}, {g, nG}],
   "A2" -> Table[{{Det[R]}}, {R, td}],
   "E" -> DE,
   "T1" -> Table[Det[R] R, {R, td}],
   "T2" -> td|>;
irreps = Keys[irrepMats];
partnerNames = <|"A1" -> {""}, "A2" -> {""}, "E" -> {"1E", "2E"}, "T1" -> {"x", "y", "z"}, "T2" -> {"x", "y", "z"}|>;
Print["Irrep matrices are unitary representations: ",
  And @@ Flatten@Table[{Simplify[ConjugateTranspose[irrepMats[ir][[i]]] . irrepMats[ir][[i]] ==
       IdentityMatrix[Length[irrepMats[ir][[1]]]]],
     Simplify[irrepMats[ir][[i]] . irrepMats[ir][[j]] == irrepMats[ir][[idx[td[[i]] . td[[j]]]]]]},
    {ir, irreps}, {i, nG}, {j, {1, 2, 5, 9, 17, 24}}]];
Print["E basis check: D^E(C3[111]) = diag(w, w^2): ", Simplify[DE[[idx[{{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}]]] == DiagonalMatrix[{w, w^2}]],
  ",  D^E(sigma_d x<->y) = swap: ", DE[[idx[{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}}]]] == {{0, 1}, {1, 0}}];

(* ---------- 4. decomposition ---------- *)
dimIr[ir_] := Length[irrepMats[ir][[1]]];
mult = AssociationMap[Simplify[Sum[Conjugate[Tr[irrepMats[#][[g]]]] Tr[UN[[g]]], {g, nG}]/nG] &, irreps];
Print["\nSym^", degree, "(T2): dimension ", dimN, " = ",
  StringRiffle[Table[If[mult[ir] == 0, Nothing, If[mult[ir] == 1, "", ToString[mult[ir]]] <> ir], {ir, irreps}], " + "]];

(* ---------- 5. matrix-valued projectors and partner-resolved SALCs ---------- *)
P[ir_, mu_, nu_] := P[ir, mu, nu] =
   Simplify[dimIr[ir]/nG Sum[Conjugate[irrepMats[ir][[g, mu, nu]]] UN[[g]], {g, nG}]];

salc[ir_] := Module[{cols, first = {}, r},
   If[mult[ir] == 0, Return[{}]];
   cols = Transpose[P[ir, 1, 1]];
   Do[r = c - Sum[f (Conjugate[f] . c), {f, first}];
    If[Norm[N[r]] > 10^-9, AppendTo[first, Simplify[r/Sqrt[Simplify[Conjugate[r] . r]]]]];
    If[Length[first] == mult[ir], Break[]], {c, cols}];
   Table[Table[Simplify[P[ir, mu, 1] . v], {mu, dimIr[ir]}], {v, first}]];
salcs = AssociationMap[salc, irreps];

(* checks: orthonormality, completeness, covariance U(R)|nu G;k> = Sum_mu D_mu nu(R)|mu G;k> *)
all = Flatten[Values[salcs], 2];
Print["SALCs orthonormal and complete: ", Simplify[Conjugate[all] . Transpose[all] == IdentityMatrix[dimN]]];
Print["SALCs transform with the irrep matrices: ",
  And @@ Flatten@Table[Simplify[UN[[g]] . salcs[ir][[k, nu]] ==
       Sum[irrepMats[ir][[g, mu, nu]] salcs[ir][[k, mu]], {mu, dimIr[ir]}]],
    {ir, Select[irreps, mult[#] > 0 &]}, {k, mult[ir]}, {nu, dimIr[ir]}, {g, nG}]];

(* ---------- 6. output ---------- *)
(* coefficients printed as  magnitude x {1, -1, I, -I, w, w^2, -w, -w^2} *)
phaseNames = {0 -> "", 1/2 -> "-", 1/4 -> "I ", 3/4 -> "-I ", 1/3 -> "w ", 2/3 -> "w^2 ", 1/6 -> "-w^2 ", 5/6 -> "-w "};
fmt[c_] := Module[{r = FullSimplify[Abs[c]], f, name},
   f = Rationalize[Mod[N[Arg[c]]/(2 Pi), 1], 10^-9];
   If[f == 1, f = 0];
   name = f /. Append[phaseNames, _ -> None];
   If[name === None, ToString[InputForm[FullSimplify[c]]],
    Which[r == 1, If[name === "", "1", StringTrim[name]],
     name === "-", "-" <> ToString[InputForm[r]],
     True, ToString[InputForm[r]] <> If[name === "", "", " " <> StringTrim[StringReplace[name, "-" -> ""]]] /. s_String :> If[StringStartsQ[name, "-"], "-" <> s, s]]]];

ket[k_] := "|" <> StringJoin[ToString /@ k] <> ">";
stateString[v_] := StringRiffle[Table[If[PossibleZeroQ[v[[i]]], Nothing,
     fmt[v[[i]]] <> " " <> ket[basisN[[i]]]], {i, dimN}], " + "] // StringReplace[#, "+ -" -> "- "] &;
Print["\nKets |n_x n_y n_z> (quanta in the Cartesian components of the T2 mode):"];
Do[
  Do[
   Do[Print[ir, If[mult[ir] > 1, " copy " <> ToString[k], ""],
     If[dimIr[ir] > 1, ", " <> partnerNames[ir][[mu]], ""], ":  ", stateString[salcs[ir][[k, mu]]]],
    {mu, dimIr[ir]}],
   {k, mult[ir]}],
  {ir, Select[irreps, mult[#] > 0 &]}];

(* everything in one place *)
TdOvertoneResult = <|"N" -> degree, "FockBasis" -> basisN, "Multiplicities" -> mult,
   "IrrepMatrices" -> irrepMats, "SALC" -> salcs|>;
Print["\nDone."];
