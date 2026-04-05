% Fact
child('prince charles', 1).
child('princess ann', 2).
child('prince andrew', 3).
child('prince edward', 4).

% Rule
older(X, Y) :- child(X, OX), child(Y, OY), OX < OY.
succession(X, Y) :- older(X, Y).

succession_line(L) :-
    findall(X, child(X,_), All), % all X such that child(X, _) is true
    tsort(All, L). % call tsort function, All = remaining nodes, [] = accumulator, L = sorted lists

% topological sort
tsort([], []). % base case
tsort(Nodes, [Node|Sorted]) :-
  select(Node, Nodes, Rest),
  \+ (member(Other, Nodes), succession(Other, Node)),
  tsort(Rest, Sorted).
