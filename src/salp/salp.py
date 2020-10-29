from src.generation.multivariate_normal import random_corelated_vectors

# TEST
# skala dla 3 rozkładów
mv = [0,1,2] # średnie
vv = [1,4,9] # wariancje
co = 0.8 # korelacja
si = 100 # ilość próbek

tmp = random_corelated_vectors(mv,vv,cor=co,size=si)

assert len(tmp) == len(mv) == len(vv) # 3 rozkłady
assert len(tmp[0]) == si # długość pierwszego (każdego) równa się size
assert tmp.size == len(mv) * si
