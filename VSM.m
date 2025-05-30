fileFolder=fullfile('G:\datasets\visible_thermal_ship\result\V'); % Folder where infrared images or visible images are located

dirOutput=dir(fullfile(fileFolder,'*.png')); % the suffix name of the source and fused images
fileNames = {dirOutput.name};
[m, num] = size(fileNames);
% ir_dir = 'F:\fusion_dataset\RoadScene_test\irRoadScene'; % Folder where infrared images are located
% vi_dir = 'F:\fusion_dataset\RoadScene_test\vi'; % Folder where visible images are locate

ir_dir = 'G:\datasets\visible_thermal_ship\result\T'; % Folder where infrared images are located
vi_dir = 'G:\datasets\visible_thermal_ship\result\V'; % Folder where visible images are located

outout_dir = 'G:\datasets\visible_thermal_ship\result\MASK'; % Folder where fused images are located
for k = 1:num
    fileName_source_ir = fullfile(ir_dir, fileNames{k})
    fileName_source_vi = fullfile(vi_dir, fileNames{k})
    ir_image = imread(fileName_source_ir);
    vi_image = imread(fileName_source_vi);
    vi_image = mean(vi_image, 3);
    
    ir_image=uint8(ir_image);
    [count, x] = imhist(ir_image);
    Sal_Tab_ir = zeros(256,1);
    for j=0:255,
        for i=0:255,
        Sal_Tab_ir(j+1) = Sal_Tab_ir(j+1)+count(i+1)*abs(j-i);    
        end      
    end
    out_ir=zeros(size(ir_image));
    for i=0:255,
        out_ir(ir_image==i)=Sal_Tab_ir(i+1);
    end 
    out_ir=mat2gray(out_ir);
    
    vi_image=uint8(vi_image);
    [count, x] = imhist(vi_image);
    Sal_Tab_vi = zeros(256,1);
    for j=0:255,
        for i=0:255,
        Sal_Tab_vi(j+1) = Sal_Tab_vi(j+1)+count(i+1)*abs(j-i);    
        end      
    end
    out_vi=zeros(size(vi_image));
    for i=0:255,
        out_vi(vi_image==i)=Sal_Tab_vi(i+1);
    end 
    out_vi=mat2gray(out_vi);
    
    weight_mask_vi = out_vi ./ (out_vi - out_ir);
    save_name = fullfile(outout_dir, fileNames{k});
    imwrite(weight_mask_vi, save_name)
    
end
