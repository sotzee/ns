#!/bin/bash
echo 'Calculating with config'

#for i in $(ls config*.py); do
#	echo $i
#	python main.py ${i/.py/} 'hybrid' 'hardest_before_trans' &> 'run_'${i}
#	python sortbymaxmass.py ${i/.py/} 'hybrid' 'hardest_before_trans' &>> 'run_'${i}
#	python main_MRdistribution.py ${i/.py/} 'hybrid' 'hardest_before_trans' &>> 'run_'${i}
#	python viewResult_sorted.py ${i/.py/} 'hybrid' 'hardest_before_trans'
#done

for i in $(ls config*.py); do
	echo $i
	python main.py ${i/.py/} 'hybrid' 'normal_trans' &> 'run_'${i}
	#python sortbymaxmass.py ${i/.py/} 'hybrid' 'normal_trans' &>> 'run_'${i}
	#python main_MRdistribution.py ${i/.py/} 'hybrid' 'normal_trans' &>> 'run_'${i}
	#python viewResult_sorted.py ${i/.py/} 'hybrid' 'normal_trans'
done

#for i in $(ls config*.py); do
#	echo $i
#	python main.py ${i/.py/} 'hybrid' 'low_trans' &> 'run_'${i}
#	python sortbymaxmass.py ${i/.py/} 'hybrid' 'low_trans' &>> 'run_'${i}
#	python main_MRdistribution.py ${i/.py/} 'hybrid' 'low_trans' &>> 'run_'${i}
#	python viewResult_sorted.py ${i/.py/} 'hybrid' 'low_trans'
#done
