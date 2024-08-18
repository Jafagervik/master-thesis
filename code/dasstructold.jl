mutable struct DASDataFrame
    datapath::String
    rows::Int
    cols::Int
    meta::Dict
    times::Vector{DateTime}
end