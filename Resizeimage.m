images = imageDatastore('./Images/Wallet',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
	for i=1:numel(images.Files)
		temp=imread(char(images.Files(i)));
		temp=imresize(temp,[227 227]);
        imwrite(temp, sprintf('Wallet_%d.jpg',i));
	end