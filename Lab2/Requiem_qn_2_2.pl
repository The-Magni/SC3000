% Fact
child('prince charles', 1).
child('princess ann', 2).
child('prince andrew', 3).
child('prince edward', 4).

% Rule
older(X, Y) :- child(X, OX), child(Y, OY), OX < OY.
succession(X, Y) :- older(X, Y).

ordered_list(L) :-
    findall(X, child(X,_), All), % all X such that child(X, _) is true
    tsort(All, [], L). % call tsort function, All = remaining nodes, [] = accumulator, L = sorted lists

tsort([], Acc, Sorted) :- reverse(Acc, Sorted). % base case: no node left to sort, reverse the order
tsort(Nodes, Acc, Sorted) :-
    % pick a node with no predecessors in Nodes
    select(Node, Nodes, Rest), % pick 1 node from Nodes, Rest = remaining node after removing
    \+ (member(Other, Nodes), succession(Other, Node)),
    tsort(Rest, [Node|Acc], Sorted).
