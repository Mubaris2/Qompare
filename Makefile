.PHONY: data prep mock test validate smoke ht tune parity run
data:     ; python data/download.py
prep:     ; python data/prep.py t1 --fit-rows 800 --margin 0.35 && python data/prep.py t2 --file data/raw/ad/23_mammography.npz --n-rows 10000 && python data/prep.py t3 --fit-rows 800 --n-rows 10000
mock:     ; python data/make_mock.py
test:     ; python -m pytest -q
validate: ; python -m common.validate_results "results/*/*.csv"
smoke:    ; python -m classical.ht_elec2 --source synth --n-rows 20000 --tag synth_smoke
ht:       ; python -m classical.ht_elec2
tune:     ; python -m classical.tune
parity:   ; python -m classical.batch_parity p2 && python -m classical.batch_parity p3 --draws 50
run:      ; python -m classical.run_experiments --workers 4
