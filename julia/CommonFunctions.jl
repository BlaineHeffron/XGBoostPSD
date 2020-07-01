function getName(indirs)
    modelname = ""
    i = 0
    for d in indirs
        if !isdir(d)
            error("Error: argument " + d + " is not a directory")
        else
            if i > 0
                modelname = string(modelname,"_" , last(splitdir(d)))
            else
                modelname = string(modelname, last(splitdir(d)))
            end
        end
        i+=1
    end
    return modelname
end
