% Fact
child('prince charles', 1).
child('princess ann', 2).
child('prince andrew', 3).
child('prince edward', 4).

% Rule
older(X, Y) :- child(X, OX), child(Y, OY), OX < OY.
succession(X, Y) :- older(X, Y).
