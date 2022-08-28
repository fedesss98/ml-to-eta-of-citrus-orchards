.PHONY: data preprocess clean

data: ~/Scaricati/Database_Castelvetrano.xlsx
	python src/data/upload.py

preprocess: data
	python src/data/preprocess.py

setup: environment.yml 
	conda env create -f environment.yml

clean: 
	rm -rf __pycache__
	rm -f data/interim/*
	rm -f data/raw/*
	rm -f data/processed/*

