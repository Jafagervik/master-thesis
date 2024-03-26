abstract type Distance end

struct Euclidean <: Distance end
struct Manhattan <: Distance end
@kwdef struct Minkowski <: Distance
    p::Int = 2
end
struct Cosine <: Distance end
struct Jaccard <: Distance end
struct Hamming <: Distance end

distance(::Euclidean, p1::Point, p2::Point) = 
    sqrt((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

distance(::Manhattan, p1::Point, p2::Point) = 
    sum(abs.(p1.x - p2.x) + abs.(p1.y - p2.y))

distance(m::Minkowski, p1::Point, p2::Point) = 
    sum(abs.(p1.x - p2.x)^m.p + abs.(p1.y - p2.y)^m.p)^(1 / m.p)

distance(::Cosine, p1::Point, p2::Point) =
    dot([p1.x, p1.y], [p2.x, p2.y]) / (norm([p1.x, p1.y]) * norm([p2.x, p2.y]))

distance(::Jaccard, p1::Point, p2::Point) =
    1 - dot([p1.x, p1.y], [p2.x, p2.y]) /
        (norm([p1.x, p1.y])^2 + norm([p2.x, p2.y])^2 
        - dot([p1.x, p1.y], [p2.x, p2.y]))

distance(::Hamming, p1::Point, p2::Point) = 
    sum(p1.x .!= p2.x) + sum(p1.y .!= p2.y)