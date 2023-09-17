function [Tr_Ind,Te_Ind]=index_5(patchLabels,trainpercentage);

Tr_Ind=[];
Te_Ind=[];

for i =1:max(patchLabels(:))
      len = length(find(patchLabels==i));    

      [id]=find(patchLabels==i);    
      randID = randperm(len);
      
      train_index = id(randID(1:trainpercentage));
      test_index= id(randID(trainpercentage+1:len));
      
      Tr_Ind=[Tr_Ind;train_index];
      Te_Ind=[Te_Ind;test_index];
         
    
end