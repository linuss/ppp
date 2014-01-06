all: asp-seq asp-par asp-par2

asp-seq: asp-seq.c
	gcc -Wall -O3 -o asp-seq asp-seq.c

asp-par: asp-par.c
	mpicc -Wall -O3 -o asp-par asp-par.c	

asp-par2: asp-par2.c
	mpicc -Wall -O3 -o asp-par2 asp-par2.c

clean:
	rm -f *.o *core *~ asp-seq asp-par asp-par2

