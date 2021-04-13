
FNAME = 'df_odor_door_all_odors.csv';
SAVETONAME = 'df_odor_door_all_odors_imput_ALS.txt';

S = vartype('numeric');
Temp = readtable(FNAME);

rng(1234);

temp = Temp{:, S};
[NGLOM, NODOR] = size(temp);
NIMPUT = 1000;

t=zeros(NGLOM, NODOR, NIMPUT);
for i=1:NIMPUT;
[coeff1,score1,latent,tsquared,explained,mu1] = pca(temp,'algorithm','als');
t(:,:,i) = score1*coeff1' + repmat(mu1,NGLOM,1);
end
t_mean=mean(t,3);
infill=t_mean(isnan(temp));
infill=[infill (1:length(infill))'];
infill=sortrows(infill,1);
infill=[infill (1:length(infill))'];
infill=sortrows(infill,2);
temp_sorted=sort(temp(~isnan(temp)));
infill_ranking=ceil(infill(:,3)*length(temp_sorted)/size(infill,1));
temp_filled=temp;temp_filled(isnan(temp))=temp_sorted(infill_ranking);

dlmwrite(SAVETONAME,temp_filled)