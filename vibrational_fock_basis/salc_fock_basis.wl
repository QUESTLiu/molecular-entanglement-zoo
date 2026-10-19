(* ::Package:: *)

(* ============================================================================
   salc_fock_basis.wl

   Symmetry-adapted (SALC) vibrational Fock basis for an arbitrary excitation pattern v,
   following the main text: Lemma 2 (Fock-space action, Eqs. (21)-(22)) and Theorem 1
   (Symmetry-adapted Fock encoding, Eqs. (23)-(25)), proved in Appendix S3 (Eqs. (S1)-(S5)).

   Input : point group, the irrep carried by each normal mode, and the quanta v_s in each mode.
   Output: (i)   characters chi_v(R) = Tr U_v(R) of the reducible representation on H(v),
           (ii)  its decomposition Gamma_red = (+)_Gamma m_Gamma(v) Gamma           [Eq. (S5)],
           (iii) the SALC states |mu Gamma; k> for every copy k = 1..m_Gamma and partner
                 mu = 1..d_Gamma, built with the matrix-valued projectors P^Gamma_{mu nu} [Eq. (25)]:
                 |1 Gamma; k> spans the range of P^Gamma_{11}, |mu Gamma; k> = P^Gamma_{mu 1}|1 Gamma; k>.

   Group elements, character table and explicit irrep matrices all come from GTPack.
   Convention (Eq. (21)): U(R) a_i^dag U(R)^dag = Sum_j D_ji(R) a_j^dag, so a single quantum
   transforms as the irrep basis vector and U(R)|nu Gamma;k> = Sum_mu D_mu nu(R) |mu Gamma;k>.

   Circular basis (circular = True): every 2D irrep is rotated so that the lowest-order proper
   rotation is diagonal with partner 1 = 1E carrying the positive phase (C3 -> diag(w, w^2)),
   and one reflection is a pure swap -- the basis used for rotation and nuclear spin in the
   main text. A quantum in partner 1 (1E) has vibrational angular momentum l = -1, so kets are
   printed in the main text's order |n+, n->, l = n+ - n-, with n- = quanta in 1E and
   n+ = quanta in 2E (U(C_n)|n+,n-> = exp(-2 pi i l/n)|n+,n->, remark after Lemma 2).
   The stored "FockLabels" keep the partner order (n_1E, n_2E).

   Run: wolframscript -file salc_fock_basis.wl          (requires GTPack)
   ============================================================================ *)

ClearAll["Global`*"];
Needs["GroupTheory`"];

(* ======================= User input ======================= *)
groupName  = "C3v";
modeIrreps = {"A1", "A1", "A1", "A1", "E", "E", "E", "E"};   (* CaOCH3: nu1 ... nu8 *)
vQuanta    = {1, 1, 1, 1, 1, 1, 1, 1};                       (* quanta in each mode *)
circular   = True;    (* 2D irreps in the main text's circular basis (see header); kets |n+,n->.
                         False keeps GTPack's own irrep basis. *)
numeric    = False;   (* True: floating point (use for large manifolds) *)
(* ========================================================== *)

simp[x_] := If[numeric, Chop[N[x], 10^-10], Simplify[ComplexExpand[x]]];

(* ---------- group data from GTPack ---------- *)
grp0 = GTInstallGroup[groupName, GOVerbose -> False];
{{classes, charTable, labels}, {grp, ireps}} =
  GTGetIreps[grp0, GOIrepNotation -> "Mulliken", GOVerbose -> False];

nameStr[x_String] := x;
nameStr[x_] := StringJoin[ToString /@ Cases[{x}, _String | _Integer | _Symbol, Infinity]];
irrepNames = nameStr /@ labels;
nG = Length[grp];
rot3 = N[GTGetMatrix[#]] & /@ grp;                  (* O(3) matrices, for proper/improper tests *)

irrepRaw = Table[If[MatrixQ[#], #, {{#}}] & /@ ireps[[i]], {i, Length[ireps]}];

(* circular basis for 2D irreps *)
adapt2D[mats_] := Module[{cand, g, vals, vecs, W, Dp, s},
  cand = Select[Range[nG],
    Det[rot3[[#]]] > 0 && Abs[Subtract @@ Eigenvalues[N[mats[[#]]]]] > 10^-8 &];
  If[cand === {}, Return[mats]];
  (* smallest rotation angle; ties -> counterclockwise about z (e.g. C3z before C3z^-1) *)
  g = First@SortBy[cand, {Round[ArcCos[Clip[(Tr[rot3[[#]]] - 1)/2, {-1, 1}]], 10^-6], -rot3[[#, 2, 1]]} &];
  {vals, vecs} = Eigensystem[mats[[g]]];
  W = Transpose[Normalize /@ vecs[[Ordering[N[Arg[vals]], All, Greater]]]];
  Dp = simp[ConjugateTranspose[W] . # . W] & /@ mats;
  s = SelectFirst[Range[nG], Det[rot3[[#]]] < 0 && Abs[N[Dp[[#, 1, 1]]]] < 10^-8 &, None];
  If[s =!= None,
    W = W . DiagonalMatrix[{1, Dp[[s, 2, 1]]}];
    Dp = simp[ConjugateTranspose[W] . # . W] & /@ mats];
  Dp];

irrepD = Table[
   If[circular && Length[irrepRaw[[i, 1]]] == 2, adapt2D[irrepRaw[[i]]], simp[irrepRaw[[i]]]],
   {i, Length[irrepRaw]}];
irrepDim = Length[First[#]] & /@ irrepD;
chiEl = Table[simp[Tr /@ irrepD[[i]]], {i, Length[irrepD]}];     (* characters per group element *)

unitaryQ = And @@ Flatten@Table[
    Max[Abs[N[ConjugateTranspose[m] . m - IdentityMatrix[Length[m]]]]] < 10^-8,
    {i, Length[irrepD]}, {m, irrepD[[i]]}];
If[! unitaryQ, Print["Warning: GTPack irrep matrices are not unitary; SALCs may not be orthonormal."]];

(* ---------- modes ---------- *)
nModes = Length[modeIrreps];
If[Length[vQuanta] != nModes, Print["Error: modeIrreps and vQuanta differ in length."]; Abort[]];
modeIdx = Table[
   If[IntegerQ[modeIrreps[[s]]], modeIrreps[[s]],
    Replace[FirstPosition[irrepNames, modeIrreps[[s]]],
     {{p_} :> p, _ :> (Print["Error: irrep ", modeIrreps[[s]], " not in ", irrepNames]; Abort[])}]],
   {s, nModes}];
modeDim = irrepDim[[modeIdx]];

(* Fock basis of one d-fold degenerate mode with n quanta, (n,0,..) first *)
focks[d_, n_] := ReverseSort[FrobeniusSolve[ConstantArray[1, d], n]];

(* matrix of R on the n-quantum manifold of a mode whose creation operators transform with M *)
fockMatrix[M_, n_] := Module[{d = Length[M], x, vars, basis, img, rules},
  vars = Array[x, d];
  basis = focks[d, n];
  Transpose@Table[
    img = Expand[Times @@ ((Transpose[M] . vars)^m)/Sqrt[Times @@ (m!)]];
    rules = Association[CoefficientRules[img, vars]];
    Table[Lookup[rules, Key[k], 0] Sqrt[Times @@ (k!)], {k, basis}],
    {m, basis}]];

fm = Table[simp@fockMatrix[irrepD[[modeIdx[[s]], g]], vQuanta[[s]]], {s, nModes}, {g, nG}];
(* U_v(R); Fold handles a single mode too (KroneckerProduct[M] alone stays unevaluated) *)
Uv = Table[simp[Fold[KroneckerProduct, {{1}}, fm[[All, g]]]], {g, nG}];
fockLabels = Tuples[Table[focks[modeDim[[s]], vQuanta[[s]]], {s, nModes}]];
dimH = Length[fockLabels];

(* ---------- reducible representation and multiplicities, Eq. (S5) ---------- *)
chiV = Table[simp[Times @@ (Tr /@ fm[[All, g]])], {g, nG}];
mult = Table[Round[N[Sum[Conjugate[chiEl[[i, g]]] chiV[[g]], {g, nG}]/nG]], {i, Length[irrepD]}];
If[mult . irrepDim != dimH, Print["Warning: sum of m d = ", mult . irrepDim, " != dim H(v) = ", dimH]];

classRep = First@FirstPosition[grp, First[#]] & /@ classes;
decomp = StringRiffle[
   Table[If[mult[[i]] == 0, Nothing,
     If[mult[[i]] == 1, "", ToString[mult[[i]]]] <> irrepNames[[i]]], {i, Length[mult]}],
   " \[CirclePlus] "];

(* ---------- SALCs from matrix-valued projectors, Theorem 1, Eq. (25) ---------- *)
proj[i_, mu_, nu_] := proj[i, mu, nu] =
   simp[irrepDim[[i]]/nG Sum[Conjugate[irrepD[[i, g, mu, nu]]] Uv[[g]], {g, nG}]];

gramSchmidt[vs_] := Module[{out = {}, w},
  Do[w = vv - Sum[u (Conjugate[u] . vv), {u, out}];
   AppendTo[out, simp[w/Sqrt[simp[Conjugate[w] . w]]]], {vv, vs}];
  out];

(* returns {copy k -> {partner mu -> vector}} *)
salc[i_] := Module[{m = mult[[i]], P1, c, r, Q = {}, keep = {}, vecs},
  If[m == 0, Return[{}]];
  P1 = proj[i, 1, 1];
  Do[
   c = P1[[All, j]];
   r = N[c] - Sum[q (Conjugate[q] . N[c]), {q, Q}];
   If[Norm[r] > 10^-8, AppendTo[Q, r/Norm[r]]; AppendTo[keep, c]];
   If[Length[keep] == m, Break[]],
   {j, dimH}];
  vecs = gramSchmidt[keep];
  Table[Table[If[mu == 1, vecs[[k]], simp[proj[i, mu, 1] . vecs[[k]]]], {mu, irrepDim[[i]]}], {k, m}]];

salcs = Table[salc[i], {i, Length[irrepD]}];

(* ---------- output ---------- *)
ra = "\[RightAngleBracket]";
nondeg = Select[Range[nModes], modeDim[[#]] == 1 &];
deg = Complement[Range[nModes], nondeg];
(* circular 2D modes: print |n+, n-> = (quanta in 2E, quanta in 1E), the main text's order *)
modeOcc[lab_, s_] := If[circular && modeDim[[s]] == 2, Reverse[lab[[s]]], lab[[s]]];
ketStr[lab_] := If[deg === {}, "|0" <> ra,
   StringJoin[("|" <> StringRiffle[ToString /@ modeOcc[lab, #], ","] <> ra) & /@ deg]];
Format[ket[s_String]] := s;
stateExpr[vec_] := Total@Table[
    If[Chop[N[vec[[j]]], 10^-10] == 0, 0, vec[[j]] ket[ketStr[fockLabels[[j]]]]], {j, dimH}];

Print["Group ", groupName, ",  |G| = ", nG, ",  irreps ", irrepNames];
Print["Modes ", Thread[modeIrreps -> vQuanta], ",  dim H(v) = ", dimH];
Print["Common factor from nondegenerate modes ", nondeg, ":  |",
  StringRiffle[ToString /@ vQuanta[[nondeg]], ","], ra];
Print["Kets list the degenerate modes ", deg,
  If[circular, " as |n+,n-> (l = n+ - n-; a 1E quantum has l = -1)", " in GTPack's irrep basis"]];

Print[Grid[
   Prepend[
    Table[Prepend[chiEl[[i, classRep]], irrepNames[[i]]], {i, Length[irrepNames]}]~Append~
     Prepend[chiV[[classRep]], "\[CapitalGamma]red"],
    Prepend[ToString[Length[classes[[#]]]] <> " " <> ToString[GTGetSymbol[classes[[#, 1]]]] & /@
      Range[Length[classes]], ""]],
   Frame -> All, Background -> {None, {LightGray}}]];
Print["\[CapitalGamma]red = ", decomp, "   (", StringRiffle[ToString /@ (mult irrepDim), " + "],
  " = ", mult . irrepDim, ")"];

Print[Grid[
   Prepend[
    Flatten[Table[{irrepNames[[i]], k, mu, stateExpr[salcs[[i, k, mu]]]},
      {i, Length[irrepD]}, {k, mult[[i]]}, {mu, irrepDim[[i]]}], 2],
    {"\[CapitalGamma]", "copy k", "partner \[Mu]", "SALC state"}],
   Frame -> All, Alignment -> Left, Background -> {None, {LightGray}}]];

(* everything in one place for further use *)
SALCresult = <|
   "Group" -> groupName, "Irreps" -> irrepNames, "IrrepMatrices" -> AssociationThread[irrepNames, irrepD],
   "Modes" -> modeIrreps, "v" -> vQuanta, "FockLabels" -> fockLabels,
   "Characters" -> AssociationThread[GTGetSymbol /@ grp, chiV],
   "Multiplicities" -> AssociationThread[irrepNames, mult], "Decomposition" -> decomp,
   "SALC" -> AssociationThread[irrepNames, salcs]|>;



