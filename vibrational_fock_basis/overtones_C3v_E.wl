(* ::Package:: *)

(* ============================================================================ *)
(*  overtones_C3v_E.wl                                                          *)
(*                                                                              *)
(*  Symmetry-adapted states of the n-quantum overtone of ONE doubly degenerate  *)
(*  E mode in C3v (e.g. nu4 of NH3, nu8 of CaOCH3), in the circular basis       *)
(*  |n+, n->, n+ + n- = n, with vibrational angular momentum l = n+ - n-.       *)
(*                                                                              *)
(*  Conventions (main text, Lemma 2 and the remark after it; Eq. tour_vib_circ):*)
(*     U(C3)      |n+,n-> = w^(-l) |n+,n->,          w = Exp[2 Pi I/3]          *)
(*     U(sigma_1) |n+,n-> = |n-,n+>                  (pure swap)                *)
(*     sigma_2 = sigma_1 C3,  sigma_3 = sigma_1 C3^2                             *)
(*  so one quantum gives |1E> = |l=-1>, |2E> = |l=+1>, and in general           *)
(*     E copies : |1E> = |l>, |2E> = |-l>   with l = 2 (mod 3)                  *)
(*     A1 / A2  : (|l> +- |-l>)/Sqrt[2] for l = 0 (mod 3), l > 0;  |l=0> is A1. *)
(*  The E partners are generated with the matrix-valued projectors              *)
(*  P^E_{mu nu} (Theorem 1, Eq. (25)) using the E matrices                       *)
(*     D(C3) = diag(w, w^2),  D(sigma_1) = {{0,1},{1,0}},                        *)
(*  i.e. the same E basis as the rotational and nuclear-spin carriers.          *)
(*                                                                              *)
(*  Run:  wolframscript -file overtones_C3v_E.wl                                *)
(* ============================================================================ *)

ClearAll["Global`*"];

(* ---------------- user input: overtone order n ---------------- *)
n = 6;
(* --------------------------------------------------------------- *)

w = Exp[2 Pi I/3];
elements = {"E", "C3", "C3^2", "sigma1", "sigma2", "sigma3"};
order = Length[elements];

(* action on |n+, n->: {phase, image} *)
act[g_, {np_, nm_}] := With[{l = np - nm},
   Switch[g,
    "E", {1, {np, nm}},
    "C3", {w^(-l), {np, nm}},
    "C3^2", {w^(-2 l), {np, nm}},
    "sigma1", {1, {nm, np}},
    "sigma2", {w^(-l), {nm, np}},       (* sigma_1 C3   *)
    "sigma3", {w^(-2 l), {nm, np}}]];   (* sigma_1 C3^2 *)

(* irrep matrices in the paper's basis *)
Dmat["A1", g_] := {{1}};
Dmat["A2", g_] := {{If[StringStartsQ[g, "sigma"], -1, 1]}};
Dmat["E", g_] := Switch[g,
   "E", IdentityMatrix[2],
   "C3", DiagonalMatrix[{w, w^2}],
   "C3^2", DiagonalMatrix[{w^2, w}],
   "sigma1", {{0, 1}, {1, 0}},
   "sigma2", {{0, 1}, {1, 0}} . DiagonalMatrix[{w, w^2}],
   "sigma3", {{0, 1}, {1, 0}} . DiagonalMatrix[{w^2, w}]];
irreps = {"A1", "A2", "E"};
dim[ir_] := Length[Dmat[ir, "E"]];

basis = Table[{np, n - np}, {np, n, 0, -1}];      (* |n,0>, |n-1,1>, ..., |0,n> *)
nb = Length[basis];

U[g_] := Module[{M = ConstantArray[0, {nb, nb}], c, img},
   Do[{c, img} = act[g, basis[[j]]];
    M[[First@FirstPosition[basis, img], j]] = c, {j, nb}];
   M];

(* consistency checks: U is a representation, D are representations *)
mulTable = Association@Flatten@Table[{g, h} -> SelectFirst[elements,
       Simplify[U[#] == U[g] . U[h]] &], {g, elements}, {h, elements}];
Print["U is a homomorphism: ", FreeQ[Values[mulTable], Missing]];
Print["E irrep matrices respect the same products: ",
  And @@ Flatten@Table[Simplify[Dmat["E", mulTable[{g, h}]] == Dmat["E", g] . Dmat["E", h]], {g, elements}, {h, elements}]];

(* multiplicities and matrix-valued projectors *)
chi[ir_, g_] := Tr[Dmat[ir, g]];
mult[ir_] := Simplify[Sum[Conjugate[chi[ir, g]] Tr[U[g]], {g, elements}]/order];
P[ir_, mu_, nu_] := Simplify[dim[ir]/order Sum[Conjugate[Dmat[ir, g][[mu, nu]]] U[g], {g, elements}]];

salc[ir_] := Module[{P11 = P[ir, 1, 1], cols, first},
   cols = Select[Transpose[P11], Norm[N[#]] > 10^-10 &];
   first = Simplify[Orthogonalize[DeleteDuplicates[cols, Norm[N[#1 - #2]] < 10^-10 &]]];
   first = Select[first, Norm[N[#]] > 10^-10 &];
   Table[Table[Simplify[P[ir, mu, 1] . v], {mu, dim[ir]}], {v, first}]];

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

ket[{np_, nm_}] := "|" <> ToString[np] <> "," <> ToString[nm] <> ">";
stateString[v_] := StringRiffle[
   Table[If[PossibleZeroQ[v[[i]]], Nothing,
     fmt[v[[i]]] <> " " <> ket[basis[[i]]] <>
      "(l=" <> ToString[basis[[i, 1]] - basis[[i, 2]]] <> ")"], {i, nb}], " + "] // StringReplace[#, "+ -" -> "- "] &;

Print["============================================================"];
Print["E-mode overtone n = ", n, "   basis |n+,n-> (l = n+ - n-): ", ket /@ basis];
Print["Decomposition: ", StringRiffle[
   Table[If[mult[ir] == 0, Nothing, If[mult[ir] == 1, "", ToString[mult[ir]]] <> ir], {ir, irreps}], " + "],
  "   (dimension ", nb, ")"];
Do[
  With[{s = salc[ir]},
   Do[
    If[dim[ir] == 1,
     Print[ir, " copy ", k, ":  ", stateString[s[[k, 1]]]],
     Print["E copy ", k, ":  |1E> = ", stateString[s[[k, 1]]], "    |2E> = ", stateString[s[[k, 2]]]]],
    {k, Length[s]}]],
  {ir, irreps}];
Print["Done."];
