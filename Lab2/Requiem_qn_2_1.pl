% Facts
child('prince charles', 1).
child('princess ann', 2).
child('prince andrew', 3).
child('prince edward', 4).

male('prince charles').
male('prince andrew').
male('prince edward').

female('princess ann').
% Rules
older(X, Y) :- child(X, OX), child(Y, OY), OX < OY.
succession(X, Y) :- male(X), female(Y).
succession(X, Y) :- male(X), male(Y), older(X, Y).
succession(X, Y) :- female(X), female(Y), older(X, Y).
