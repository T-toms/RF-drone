s = load('G:\drone\raw\bg_feature_cache.mat');
out = struct();
out.fields = fieldnames(s);
out.entries = {};
for fi = 1:numel(out.fields)
    obj = s.(out.fields{fi});
    if isa(obj, 'containers.Map')
        kk = keys(obj);
        out.map_count = numel(kk);
        for ki=1:numel(kk)
            v = obj(kk{ki});
            e = struct('key',kk{ki},'class',class(v),'size',size(v));
            if isnumeric(v)
                e.minimum = min(v(:)); e.maximum=max(v(:));
            elseif isstruct(v)
                e.fields=fieldnames(v); e.details=struct();
                for vi=1:numel(e.fields)
                    q=v.(e.fields{vi});
                    z=struct('class',class(q),'size',size(q));
                    if isnumeric(q) && numel(q)>0
                        z.minimum=min(q(:));z.maximum=max(q(:));
                        if numel(q)<=16,z.values=q;end
                    elseif ischar(q) || isstring(q),z.values=q;
                    end
                    e.details.(e.fields{vi})=z;
                end
            end
            out.entries{end+1}=e;
        end
    end
end
fid=fopen(fullfile(fileparts(mfilename('fullpath')),'audit_drone_20261023','bg_cache_decoded.json'),'w','n','UTF-8');
fwrite(fid,jsonencode(out),'char');fclose(fid);
disp(out.fields);disp(out.map_count);
