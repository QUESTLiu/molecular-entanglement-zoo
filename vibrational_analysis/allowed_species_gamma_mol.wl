(* ::Package:: *)

(* Plain-text export of allowed_species_gamma_mol.nb (same code, no outputs).
   Admissible (Gamma_rot, Gamma_nuc_spin, Gamma_vib) triples containing Gamma_mol, by character
   decomposition, for C2v (B2), C3v (A2, with the explicit coupled states as LaTeX), Td (A2) and Ih (Ag).
   Requires GTPack (Needs["GroupTheory`"]); the Ih block uses hard-coded characters. *)

Needs["GroupTheory`"];

ClearAll[MakeTableData, DecomposeProduct, AllowedTriples];

MakeTableData[pointGroup_, irrepLabels_] := 
  Module[{g, raw, classes, chars, classSizes, nChars, nLabels}, 
   g = GTInstallGroup[pointGroup, GOVerbose -> False];
   raw = 
    GTCharacterTable[g, GOIrepNotation -> "Mulliken", 
     GOVerbose -> False];
   classes = raw[[1]];
   chars = raw[[2]];
   classSizes = Length /@ classes;
   nChars = Length[chars];
   nLabels = Length[irrepLabels];
   If[nChars =!= nLabels, 
    Print["Error: The point group ", pointGroup, " has ", nChars, 
     " irreps, but you provided ", nLabels, " labels."];
    Return[$Failed];];
   <|"PointGroup" -> pointGroup, "Labels" -> irrepLabels, 
    "ClassSizes" -> classSizes, "Order" -> Total[classSizes], 
    "Characters" -> AssociationThread[irrepLabels -> chars]|>];

DecomposeProduct[data_, reps_List] := 
  Module[{labels, chars, classSizes, order, productChar, mults}, 
   labels = data["Labels"];
   chars = data["Characters"];
   classSizes = data["ClassSizes"];
   order = data["Order"];
   productChar = Times @@ (chars[#] & /@ reps);
   mults = 
    AssociationThread[labels, 
     Simplify[
      Table[Total[classSizes*Conjugate[chars[label]]*productChar]/
        order, {label, labels}]]];
   Select[mults, # != 0 &]];

AllowedTriples[data_, gammaRotList_, gammaNucList_, gammaVibList_, 
   gammaMol_] := 
  Module[{triples, rows, decomposition, multiplicity}, 
   triples = Tuples[{gammaRotList, gammaNucList, gammaVibList}];
   rows = Reap[Do[decomposition = DecomposeProduct[data, triple];
       multiplicity = Lookup[decomposition, gammaMol, 0];
       If[multiplicity > 0, 
        Sow[<|"Gamma_rot" -> triple[[1]], 
          "Gamma_nuc_spin" -> triple[[2]], "Gamma_vib" -> triple[[3]],
           "Gamma_total" -> decomposition, 
          "Contains_Gamma_mol" -> True, 
          "Multiplicity" -> multiplicity|>]], {triple, triples}]][[2]];
   If[rows === {}, Dataset[{}], Dataset[First[rows]]]];

ClearAll[TexIrrep, C3vA2Basis, AddBasisColumn];

TexIrrep["A1"] := "A_1";
TexIrrep["A2"] := "A_2";
TexIrrep["E"] := "E";

C3vA2Basis[rot_, spin_, vib_] := 
  Which[rot == "A1" && spin == "A1" && vib == "A2", 
   "\\ket{A_1}_{\\rm rot}\\ket{A_1}_{\\rm spin}^{(\\rm \
nuc)}\\ket{A_2}_{\\rm vib}", 
   rot == "A1" && spin == "A2" && vib == "A1", 
   "\\ket{A_1}_{\\rm rot}\\ket{A_2}_{\\rm spin}^{(\\rm \
nuc)}\\ket{A_1}_{\\rm vib}", 
   rot == "A2" && spin == "A1" && vib == "A1", 
   "\\ket{A_2}_{\\rm rot}\\ket{A_1}_{\\rm spin}^{(\\rm \
nuc)}\\ket{A_1}_{\\rm vib}", 
   rot == "A2" && spin == "A2" && vib == "A2", 
   "\\ket{A_2}_{\\rm rot}\\ket{A_2}_{\\rm spin}^{(\\rm \
nuc)}\\ket{A_2}_{\\rm vib}", 
   MemberQ[{"A1", "A2"}, rot] && spin == "E" && vib == "E", 
   "\\frac{1}{\\sqrt{2}}\\ket{" <> TexIrrep[rot] <> "}_{\\rm rot}
  \\left(
  \\ket{^{1}E}_{\\rm spin}^{(\\rm nuc)}\\ket{^{-}E}_{\\rm vib}
  +(-1)^j
  \\ket{^{2}E}_{\\rm spin}^{(\\rm nuc)}\\ket{^{+}E}_{\\rm vib}
  \\right)", rot == "E" && MemberQ[{"A1", "A2"}, spin] && vib == "E", 
   "\\frac{1}{\\sqrt{2}}\\ket{" <> TexIrrep[spin] <> 
    "}_{\\rm spin}^{(\\rm nuc)}
  \\left(
  \\ket{^{+}E}_{\\rm vib}\\ket{^{2}E}_{\\rm rot}
  +(-1)^j
  \\ket{^{-}E}_{\\rm vib}\\ket{^{1}E}_{\\rm rot}
  \\right)", rot == "E" && spin == "E" && MemberQ[{"A1", "A2"}, vib], 
   "\\frac{1}{\\sqrt{2}}\\ket{" <> TexIrrep[vib] <> "}_{\\rm vib}
  \\left(
  \\ket{^{1}E}_{\\rm spin}^{(\\rm nuc)}\\ket{^{2}E}_{\\rm rot}
  +(-1)^j
  \\ket{^{2}E}_{\\rm spin}^{(\\rm nuc)}\\ket{^{1}E}_{\\rm rot}
  \\right)", rot == "E" && spin == "E" && vib == "E", 
   "\\frac{1}{\\sqrt{2}}
  \\left(
  \\ket{^{1}E}^{(\\rm nuc)}_{\\rm spin}
  \\ket{^{1}E}_{\\rm rot}
  \\ket{^{+}E}_{\\rm vib}
  -
  \\ket{^{2}E}^{(\\rm nuc)}_{\\rm spin}
  \\ket{^{2}E}_{\\rm rot}
  \\ket{^{-}E}_{\\rm vib}
  \\right)", True, ""];

AddBasisColumn[dataset_] := 
  Dataset[Append[#, 
      "Basis" -> 
       C3vA2Basis[#["Gamma_rot"], #["Gamma_nuc_spin"], #[
         "Gamma_vib"]]] & /@ Normal[dataset]];

 (* C2v group calculation with Gamma_mol=B2 *)

Clear[data, gammaRotList, gammaNucList, gammaVibList, gammaMol];

data = MakeTableData["C2v", {"A1", "A2", "B1", "B2"}];

gammaRotList = {"A1", "B2"};
gammaNucList = {"A1", "B2"};
gammaVibList = {"A1", "B2"};

gammaMol = "B2";

AllowedTriples[data, gammaRotList, gammaNucList, gammaVibList, \
gammaMol]

(* C3v group calculation with Gamma_mol=A2 *)
Clear[data, gammaRotList, gammaNucList, gammaVibList, gammaMol];

data = MakeTableData["C3v", {"A1", "A2", "E"}];

gammaRotList = {"A1", "A2", "E"};
gammaNucList = {"A1", "A2", "E"};
gammaVibList = {"A1", "A2", "E"};

gammaMol = "A2";

AllowedTriples[data, gammaRotList, gammaNucList, gammaVibList, \
gammaMol]

data = MakeTableData["C3v", {"A1", "A2", "E"}];

gammaRotList = {"A1", "A2", "E"};
gammaNucList = {"A1", "A2", "E"};
gammaVibList = {"A1", "A2", "E"};

gammaMol = "A2";

allowed = 
  AllowedTriples[data, gammaRotList, gammaNucList, gammaVibList, 
   gammaMol];

AddBasisColumn[allowed]

(* Td group calculation with Gamma_mol = A2 *)
Clear[data, gammaRotList, gammaNucList, gammaVibList, gammaMol];

data = MakeTableData["Td", {"A1", "A2", "E", "T1", "T2"}];

gammaRotList = {"A1", "A2", "E", "T1", "T2"};
gammaNucList = {"A1", "A2", "E", "T1", "T2"};
gammaVibList = {"A1", "A2", "E", "T1", "T2"};

gammaMol = "A2";

AllowedTriples[data, gammaRotList, gammaNucList, gammaVibList, \
gammaMol]

(* Ih group calculation with Gamma_mol = A2 *)
ClearAll[MakeIhData, DecomposeProduct, AllowedTriples];

MakeIhData[] := 
  Module[{phi, psi, labels, classSizes, charsI, charsIh}, 
   phi = (1 + Sqrt[5])/2;
   psi = (1 - Sqrt[5])/2;
   labels = {"Ag", "T1g", "T2g", "Gg", "Hg", "Au", "T1u", "T2u", "Gu",
      "Hu"};
   (*Classes:E,12 C5,12 C5^2,20 C3,15 C2,i,12 iC5,12 iC5^2,20 iC3,
   15 iC2*)classSizes = {1, 12, 12, 20, 15, 1, 12, 12, 20, 15};
   (*Rotational icosahedral group I irreps:A,T1,T2,G,H*)
   charsI = {{1, 1, 1, 1, 1}, {3, phi, psi, 0, -1}, {3, psi, phi, 
      0, -1}, {4, -1, -1, 1, 0}, {5, 0, 0, -1, 1}};
   (*Ih=I x Ci.g irreps are even under inversion;
   u irreps are odd under inversion.*)
   charsIh = 
    Join[Table[Join[charsI[[k]], charsI[[k]]], {k, Length[charsI]}], 
     Table[Join[charsI[[k]], -charsI[[k]]], {k, Length[charsI]}]];
   <|"PointGroup" -> "Ih", "Labels" -> labels, 
    "ClassSizes" -> classSizes, "Order" -> Total[classSizes], 
    "Characters" -> AssociationThread[labels -> charsIh]|>];

DecomposeProduct[data_, reps_List] := 
  Module[{labels, chars, classSizes, order, productChar, mults}, 
   labels = data["Labels"];
   chars = data["Characters"];
   classSizes = data["ClassSizes"];
   order = data["Order"];
   productChar = Times @@ (chars[#] & /@ reps);
   mults = 
    AssociationThread[labels, 
     Simplify[
      Table[Total[classSizes*Conjugate[chars[label]]*productChar]/
        order, {label, labels}]]];
   Select[mults, # != 0 &]];

AllowedTriples[data_, gammaRotList_, gammaNucList_, gammaVibList_, 
   gammaMol_] := 
  Module[{triples, rows, decomposition, multiplicity}, 
   triples = Tuples[{gammaRotList, gammaNucList, gammaVibList}];
   rows = Reap[Do[decomposition = DecomposeProduct[data, triple];
       multiplicity = Lookup[decomposition, gammaMol, 0];
       If[multiplicity > 0, 
        Sow[<|"Gamma_rot" -> triple[[1]], 
          "Gamma_nuc_spin" -> triple[[2]], "Gamma_vib" -> triple[[3]],
           "Gamma_total" -> decomposition, 
          "Multiplicity" -> multiplicity|>]], {triple, triples}]][[2]];
   If[rows === {}, Dataset[{}], Dataset[First[rows]]]];

data = MakeIhData[];

gammaRotList = {"Ag", "T1g", "T2g", "Gg", "Hg"};
gammaNucList = {"Ag", "T1g", "T2g", "Gg", "Hg", "Au", "T1u", "T2u", 
   "Gu", "Hu"};
gammaVibList = {"Ag", "T1g", "T2g", "Gg", "Hg", "Au", "T1u", "T2u", 
   "Gu", "Hu"};

gammaMol = "Ag";

AllowedTriples[data, gammaRotList, gammaNucList, gammaVibList, \
gammaMol]