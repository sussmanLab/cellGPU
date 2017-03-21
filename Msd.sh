#!/bin/bash

n=5000
i=10000
t=100000

v=0.1
e=0.1
g=0



d3=1.0
for p in 3.5 3.6 3.65 3.7 3.75 3.8 3.85
do
./spvMSD.out "-n" $n "-v" $v "-e" $e "-i" $i "-t" $t "-d" $d3 "-p" $p  "-g" $g
done
