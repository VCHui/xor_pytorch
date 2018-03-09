set term x11
set xlabel "A"
set ylabel "B"
set xrange [-0.5:1.5]
set yrange [-0.5:1.5]
set xtics (0,1)
set ytics (0,1)
f(t) = 1/(1+exp(-t)) # logistic function
a = 20
t0(x,y) = f(a*(f(a*(x-y-0.75))-f(a*(x-y+0.75))+0.5))
t1(x,y) = f(a*(f(a*(x+y-1.75))-f(a*(x+y-0.25))+0.5))
