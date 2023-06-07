digt="1 2 3 4 5 6 7 8 9 0"
name="jackson nicolas theo"
for i in $digt
	do
		for n in $name
			do
				ls train/$i | grep $n | head -n 10 | xargs -i mv train/$i/{} test/$i
			done
		
	done
exit 0