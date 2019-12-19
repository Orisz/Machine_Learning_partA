function [TestSetError] = runTree(TestSet, TestLabels, Tree)
%runTree - classify each person in ValidSet according to Tree

[D, n] = size(TestSet);
TestTreeTags = zeros(n,1);
for i=1:n
    curr_node=1; %start at root
    while(Tree.nodes(curr_node,1) ~= -1) % didn't arrive at a leaf yet
        curr_thresh = Tree.nodes(curr_node,2);
        curr_D = Tree.nodes(curr_node,1);
        if (TestSet(curr_D,i) <= curr_thresh) %go left
            curr_node = curr_node*2;
        else %go right
            curr_node = curr_node*2 + 1;
        end
    end
    TestTreeTags(i) = Tree.nodes(curr_node,2); %save classification tag
end

TestSetError = mean(abs(TestLabels - TestTreeTags));

end

