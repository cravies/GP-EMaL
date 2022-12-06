wine:
	#run it!
	echo "Running wine dataset, 30 gens, time objective."
	python3 -m gpmalmo.gpmal_nc -d wine --dir "datasets/" --gens 100 --obj time
	echo "Running wine dataset, 30 gens, size objective."
	python3 -m gpmalmo.gpmal_nc -d wine --dir "datasets/" --gens 100 --obj size
clean:
	#remove old log files
	for filetype in png csv tree gz; do \
		rm *.$$filetype || echo "no ."$$filetype" found" ; \
	done