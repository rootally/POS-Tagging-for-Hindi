#!/bin/bash
rm fin
for file in *.txt;do
	cat $file | grep -E ".\.." > tmp
	cat tmp | cut -c 1,4- > tmp2
	prev=0
	while read -r line;do
		num=`echo $line | cut -c 1`
		tmp=`echo $line | awk '{print $2 "/" $3}'`
        	if [ $num -ge $prev ];then
			echo -n "$tmp " >> fin
			prev=$num
		else
			prev=$num
			echo -ne "\n$tmp " >> fin
		fi	
	done < tmp2
	echo "" >> fin
done
rm tmp2 tmp
