#pragma once

template <typename A, typename Y> struct Game {

    virtual double loss(const A &a, const Y &y) const = 0;
};
