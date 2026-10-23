(* Independent check of salc_tables.py (multiplicities) for the 32 crystallographic point groups, using
   GTPack character tables. chi_{Sym^v}(g) from power sums p_k(g) = chi(g^k) via Newton's identity; g^k is
   located with the faithful 3x3 O(3) matrices of GTPack. Writes multiplicities_gtpack.json. *)
Needs["GroupTheory`"];
vmax = 12;
groups = {"C1", "Ci", "C2", "Cs", "C2h", "D2", "C2v", "D2h", "C4", "S4", "C4h", "D4", "C4v", "D2d", "D4h",
   "C3", "S6", "D3", "C3v", "D3d", "C6", "C3h", "C6h", "D6", "C6v", "D3h", "D6h", "T", "Th", "O", "Td", "Oh"};
nameStr[x_String] := x;
nameStr[x_] := StringJoin[ToString /@ Cases[{x}, _String | _Integer | _Symbol, Infinity]];
out = <||>;
Do[
  grp0 = GTInstallGroup[gname, GOVerbose -> False];
  {classes, charTable, labels} = GTCharacterTable[grp0, GOVerbose -> False, GOIrepNotation -> "Mulliken"];
  grp = Flatten[classes, 1];
  nG = Length[grp];
  rot3 = N[GTGetMatrix[#]] & /@ grp;
  clsOf = Flatten[Table[ConstantArray[c, Length[classes[[c]]]], {c, Length[classes]}]];
  idxOf[mm_] := SelectFirst[Range[nG], Max[Abs[rot3[[#]] - mm]] < 10^-6 &];
  powCls = Table[clsOf[[idxOf[MatrixPower[rot3[[g]], k]]]], {g, nG}, {k, vmax}];
  chi = N[charTable];                                    (* [irrep][class] *)
  nIr = Length[chi];
  dims = Round[Re[chi[[All, 1]]]];
  names = nameStr /@ labels;
  modes = {};
  Do[
   If[dims[[i]] >= 2, AppendTo[modes, {names[[i]], chi[[i]]}],
    If[Max[Abs[Im[chi[[i]]]]] > 10^-8,
     j = SelectFirst[Range[nIr], Max[Abs[chi[[#]] - Conjugate[chi[[i]]]]] < 10^-8 &];
     If[j > i, AppendTo[modes, {names[[i]] <> "+" <> names[[j]], chi[[i]] + chi[[j]]}]]]],
   {i, nIr}];
  modeOut = Table[
    hh = Table[
      p = Table[m[[2, powCls[[g, k]]]], {k, vmax}];
      h = {1.};
      Do[AppendTo[h, Sum[p[[k]] h[[vv - k + 1]], {k, vv}]/vv], {vv, vmax}];
      h, {g, nG}];
    coeffs = Table[Round[Re[Sum[Conjugate[chi[[i, clsOf[[g]]]]] hh[[g, vv + 1]], {g, nG}]/nG]], {vv, 0, vmax}, {i, nIr}];
    If[! FreeQ[coeffs, Round], Print["BAD ", gname, " ", m[[1]]]];
    <|"mode" -> m[[1]], "dim" -> Round[Re[m[[2, 1]]]], "coeffs" -> coeffs|>, {m, modes}];
  out[gname] = <|"order" -> nG, "irreps" -> names, "dims" -> dims, "modes" -> modeOut|>;
  Print[gname, " ", nG, " ", names, " modes: ", #["mode"] & /@ modeOut],
  {gname, groups}];
Export["multiplicities_gtpack.json", out, "JSON"];
Print["done"];
