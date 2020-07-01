struct Dataset
    name::String
    eventmap::Dict{String,Tuple{Int64,Int64}}
end

function addFile(d::Dataset,f::String)
    if !haskey(d.eventmap,f)
        d.eventmap[f] = (0,0)
    end
end

function setEventRange(d::Dataset,f::String,l::Tuple{Int64,Int64})
    d.eventmap[f] = l
end

function print(d::Dataset)
    println(d.name)
    for key in d.eventmap
        println(String(key,": events ", d.eventmap[key][1], " - ",d.eventmap[key][2]))
    end
end
