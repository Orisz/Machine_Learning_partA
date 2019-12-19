function [minQ_D,minThresh] = calcMinQ_D(dividedTrainSetD,dividedTrainLabel,CriteriaType)
%calculate according to CriteriaType the criteriaMat which holds the
%minimal Q(D) that's given by the specific feature and the threshold
%dividedTrainSetD: 1xn where n depends where we are in the tree. values of
%specific D
%CriteriaType: 1 = Label-Error , 2 = Gini-index , 3 = Entropy

n = length(dividedTrainLabel);
minThresh = 0;
minQ_D = inf;

if(n==2) %we get here only if we're left with two people labeled differently
    minThresh = min(dividedTrainSetD);
    minQ_D = 1;
else
    for i=1:n
       RightIdx = find(dividedTrainSetD > dividedTrainSetD(i));
       LeftIdx = find(dividedTrainSetD <= dividedTrainSetD(i));
       numOfRight = length(RightIdx);
       numOfLeft = length(LeftIdx);
       RightNumOfZeroTag = length(find(dividedTrainLabel(RightIdx) == 0));
       LeftNumOfZeroTag = length(find(dividedTrainLabel(LeftIdx) == 0));
       RightNumOfOneTag = numOfRight - RightNumOfZeroTag;
       LeftNumOfOneTag = numOfLeft - LeftNumOfZeroTag;

       if(CriteriaType == 1) %Label-Error
           if (RightNumOfOneTag < RightNumOfZeroTag) 
               RightLE = RightNumOfOneTag;
           else
               RightLE = RightNumOfZeroTag;
           end

           if (LeftNumOfOneTag < LeftNumOfZeroTag) 
               LeftLE = LeftNumOfOneTag;
           else
               LeftLE = LeftNumOfZeroTag;
           end

           Q_D = (numOfRight/n)*RightLE + (numOfLeft/n)*LeftLE;
           %update minimal Q(D)
           if (Q_D < minQ_D)
               minQ_D = Q_D;
               minThresh = dividedTrainSetD(i);
           end

       elseif(CriteriaType == 2) %Gini-index
           RightGini = 2*(RightNumOfOneTag/numOfRight)*(RightNumOfZeroTag/numOfRight);
           LeftGini = 2*(LeftNumOfOneTag/numOfLeft)*(LeftNumOfZeroTag/numOfLeft);

           Q_D = (numOfRight/n)*RightGini + (numOfLeft/n)*LeftGini;
           %update minimal Q(D)
           if (Q_D < minQ_D)
               minQ_D = Q_D;
               minThresh = dividedTrainSetD(i);
           end

       else %Entropy
           if(RightNumOfOneTag == 0 || RightNumOfZeroTag == 0)
               RightEn = 0;
               LeftEn = -(LeftNumOfOneTag/numOfLeft)*log2(LeftNumOfOneTag/numOfLeft) - (LeftNumOfZeroTag/numOfLeft)*log2(LeftNumOfZeroTag/numOfLeft); 
           elseif(LeftNumOfOneTag == 0 || LeftNumOfZeroTag == 0)
               LeftEn = 0;
               RightEn = -(RightNumOfOneTag/numOfRight)*log2(RightNumOfOneTag/numOfRight) - (RightNumOfZeroTag/numOfRight)*log2(RightNumOfZeroTag/numOfRight); 
           else
               RightEn = -(RightNumOfOneTag/numOfRight)*log2(RightNumOfOneTag/numOfRight) - (RightNumOfZeroTag/numOfRight)*log2(RightNumOfZeroTag/numOfRight); 
               LeftEn = -(LeftNumOfOneTag/numOfLeft)*log2(LeftNumOfOneTag/numOfLeft) - (LeftNumOfZeroTag/numOfLeft)*log2(LeftNumOfZeroTag/numOfLeft); 
           end

           Q_D = (numOfRight/n)*RightEn + (numOfLeft/n)*LeftEn;
           %update minimal Q(D)
           if (Q_D < minQ_D)
               minQ_D = Q_D;
               minThresh = dividedTrainSetD(i);
           end
       end
    end
end

end

