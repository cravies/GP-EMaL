run:
	echo "Running dataset $(DATASET), $(GENS) gens, $(OBJ) objective."
	python3 -m gpmalmo.gpmal_nc -d $(DATASET) --dir "datasets/" --gens $(GENS) --obj $(OBJ)
	rm -rf $(DATASET)_$(GENS)_$(OBJ) || echo "run data not found."
	mkdir $(DATASET)_$(GENS)_$(OBJ)
	mv *.png $(DATASET)_$(GENS)_$(OBJ)
	tar czf $(DATASET)_$(GENS)_$(OBJ).tar.gz $(DATASET)_$(GENS)_$(OBJ)
clean:
	#remove old log files
	for filetype in png csv tree; do \
		rm *.$$filetype || echo "no ."$$filetype" found" ; \
	done
test:
	python3 -m unittest gptools.test_gp_util
