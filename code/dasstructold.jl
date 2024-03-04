mutable struct DASDataFrame
    datapath::String
    rows::Int
    cols::Int
    meta::Dict
    times::Vector{DateTime}

    function DASDataFrame(
        dp, rs, cs, i, m
    )
        cols = length(cs)
        df = new(dp, rs, cs, m, i)

        isempty(meta) || 
        _set_dimension_range!(
            df, rs, cs)

        return df
    end
end