iris:
	#run it!
	echo "Running iris dataset, 50 gens, time objective."
	python3 -m gpmalmo.gpmal_nc -d iris --dir "datasets/" --gens 5 --obj time
	echo "Running iris dataset, 50 gens, size objective."
	python3 -m gpmalmo.gpmal_nc -d iris --dir "datasets/" --gens 5 --obj size
clean:
	#remove old log files
	for filetype in png csv tree gz; do \
		rm *.$$filetype || echo "no ."$$filetype" found" ; \
	done
