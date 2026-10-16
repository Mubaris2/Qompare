.PHONY: data prep mock test validate smoke ht
data:     ; python data/download.py
prep:     ; python data/prep.py t1 --fit-rows 4000 && python data/prep.py t2 --n-rows 10000 && python data/prep.py t3 --n-rows 10000 --fit-rows 7500
mock:     ; python data/make_mock.py
test:     ; python -m pytest -q
validate: ; python -m common.validate_results "results/*/*.csv"
smoke:    ; python -m classical.ht_elec2 --source synth --n-rows 20000 --tag synth_smoke
ht:       ; python -m classical.ht_elec2
