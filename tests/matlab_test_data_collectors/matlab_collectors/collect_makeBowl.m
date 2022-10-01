all_params = { ...
    {[40, 40, 40], [20, 20, 20], 5, 7, [23, 23, 23], 'Binary', false, 'RemoveOverlap', false}, ...
    {[40, 40, 40], [20, 20, 20], 5, 7, [23, 23, 23], 'Binary', true, 'RemoveOverlap', false}, ...
    {[40, 40, 40], [20, 20, 20], inf, 7, [23, 23, 23], 'Binary', true, 'RemoveOverlap', false}, ...
    {[40, 40, 40], [20, 20, 20], inf, 7, [23, 23, 23], 'Binary', true, 'RemoveOverlap', true}, ...
    {[40, 40, 40], [30, 35, 38], 3, 5, [17, 18, 5], 'Binary', false, 'RemoveOverlap', true}, ...
    {[40, 40, 40], [32, 35, 38], 10, 11, [17, 18, 5], 'Binary', false, 'RemoveOverlap', true}, ...
    {[40, 40, 40], [5, 35, 38], 10, 11, [17, 18, 5], 'Binary', false, 'RemoveOverlap', true}, ...
    {[40, 40, 40], [32, 2, 3], 10, 11, [17, 18, 5], 'Binary', false, 'RemoveOverlap', true}, ...
}; 

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
     
    bowl = makeBowl(params{:});
    
    idx_padded = sprintf('%06d', idx - 1);
    filename = ['collectedValues_makeBowl/' idx_padded '.mat'];
    save(filename, 'params', 'bowl');
end
