% Facts
boss_of(stevey, appy).
competitor(sumsum, appy).
developed(sumsum, galacticas3).
smart_phone_tech(galacticas3).
stole(stevey, galacticas3).

% Rules
rival(X, Y) :- competitor(X, Y).
business(P) :- smart_phone_tech(P).

unethical(X) :-
    boss_of(X, A),
    rival(B, A),
    developed(B, P),
    business(P),
    stole(X, P).