run:
	echo "Running dataset $(DATASET), $(GENS) gens, $(OBJ) objective."
	python3 -m gpmalmo.gpmal_nc -d $(DATASET) --dir "datasets/" --gens $(GENS) --obj $(OBJ)
	rm -rf pareto_front_$(DATASET)_$(GENS)_$(OBJ) || echo "pareto front not found"
	mkdir pareto_front_$(DATASET)_$(GENS)_$(OBJ)
	mv *.png pareto_front_$(DATASET)_$(GENS)_$(OBJ)
	tar czf pareto_front_$(DATASET)_$(GENS)_$(OBJ).tar.gz pareto_front_$(DATASET)_$(GENS)_$(OBJ)
clean:
	#remove old log files
	for filetype in png csv tree gz; do \
		rm *.$$filetype || echo "no ."$$filetype" found" ; \
	done
test:
	python3 -m unittest gptools.test_gp_util
