@inline function normalise(x::AbstractArray; dims=ndims(x)
        , eps=ofeltype(x, 1e-5))
  mu = mean(x, dims=dims)
  sigma = std(x, dims=dims, mean=mu, corrected=false)
  return @. (x - mu) / (sigma + eps)
end