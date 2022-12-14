run:
	#run it!
	#echo "Running iris dataset, 10 gens, tik objective."
	#python3 -m gpmalmo.gpmal_nc -d iris --dir "datasets/" --gens 100 --obj tik
	#echo "Running iris dataset, 10 gens, time objective."
	python3 -m gpmalmo.gpmal_nc -d dermatology --dir "datasets/" --gens 1 --obj time
	#echo "Running iris dataset, 10 gens, size objective."
	python3 -m gpmalmo.gpmal_nc -d dermatology --dir "datasets/" --gens 1 --obj size
clean:
	#remove old log files
	for filetype in png csv tree gz; do \
		rm *.$$filetype || echo "no ."$$filetype" found" ; \
	done
test:
	python3 -m unittest gptools.test_gp_util
