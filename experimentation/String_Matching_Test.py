import Levenshtein
from rapidfuzz import fuzz, utils, process
import re

control_input = ["Tom Cruise", "Brad Pitt", "Leonardo DiCaprio"]
test_input_1 = ["tom", "brad", "leonardo"]
test_input_2 = ["tom criuse", "braf pittt", "leo dicaprip"]
test_input_3 = ["tom john", "bad pitchd", "leondoff dcropio"]
test_input_4 = ["Tom Hanks and Brad Pitt standing", "tom and chad standing", "leonardo and brad sitting in the snow"]

test_inputs = [test_input_1, test_input_2, test_input_3, test_input_4]

for i in range(4):
    test_input = test_inputs[i]
    for j in range(3):
        cntrl = control_input[j]
        tst = test_input[j]

        # levenshtein Values
        l_d = Levenshtein.distance(cntrl, tst)
        l_r = Levenshtein.ratio(cntrl, tst)
        l_j = Levenshtein.jaro(cntrl, tst)
        l_jw = Levenshtein.jaro_winkler(cntrl, tst)

        # Fuzz Values
        f_r = fuzz.ratio(cntrl, tst)
        f_pr = fuzz.partial_ratio(cntrl, tst)
        f_tsr = fuzz.token_set_ratio(cntrl, tst)
        f_wr = fuzz.WRatio(cntrl, tst, processor=utils.default_process)

        print("_________________________________________________")
        print(f"Calculations between {cntrl} and {tst} :")
        print("_________________________________________________")
        print(f"Levenshtein Distance : {l_d}")
        print(f"Levenshtein Ratio : {l_r}")
        print(f"Levenshtein Jaro Distance : {l_d}")
        print(f"Levenshtein Jaro Winkler Distance : {l_r}")
        print("_________________________________________________")
        print(f"Fuzz Ratio : {f_r}")
        print(f"Fuzz Partial Ratio : {f_pr}")
        print(f"Fuzz Token Set Ratio : {f_tsr}")
        print(f"Fuzz Weighted Ratio : {f_wr}")
        print("_________________________________________________")


ref_emb = ['Tom Hanks', 'Tom Cruise', 'Brad Pitt', 'Leo Dicaprio', 'Angelina Jolie']
query = "leo, tom h, tommy, angie and brad standing"

names = re.split(r',|\band\b', query)  # Split by commas and 'and'
names = [name.strip() for name in names if name]  # Remove spaces


matched = [process.extractOne(name, ref_emb, scorer=fuzz.WRatio)[0] for name in names]
print(matched)