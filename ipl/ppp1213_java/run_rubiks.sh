#!/bin/sh

ADDRESS=$1
TWISTS=$2
S=$3


#P=1
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=2
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=3
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=4
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=5
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=6
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=7
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
P=8
prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
P=9
prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=10
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
#P=16
#prun -v -1 -np $P ./bin/java-run -Dibis.pool.name=PPP -Dibis.server.address=10.149.0.$ADDRESS -Dibis.pool.size=$P rubiks.ipl.Rubiks --twists $TWISTS  --seed $S
