Base.@kwdef mutable struct RobustPardisoMKLOptions <: MadNLP.AbstractOptions
    pardisomkl_num_threads::Int = 1
    pardisomkl_max_iterative_refinement_steps::Int = 2
    pardisomkl_msglvl::Int = 0
    pardisomkl_order::Int = 2
end

"""
PARDISO MKL variant for indefinite KKT systems with conservative numerical
defaults. Factorization, solve, inertia, and cleanup remain delegated to the
upstream MadNLPPardiso implementation.
"""
mutable struct RobustPardisoMKLSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::MadNLPPardiso.PardisoMKLSolver{T}
end

function RobustPardisoMKLSolver(
    csc::MadNLP.SparseMatrixCSC{T};
    opt = RobustPardisoMKLOptions(),
    logger = MadNLP.MadNLPLogger(),
) where {T}
    w = Vector{T}(undef, csc.n)
    pt = fill(Ptr{Cvoid}(C_NULL), 64)
    iparm = Vector{Int32}(undef, 64)
    perm = Vector{Int32}(undef, csc.n)
    msglvl = Ref{Int32}(opt.pardisomkl_msglvl)
    err = Ref{Int32}(0)

    iparm[1] = 0
    MadNLPPardiso.pardisomkl_pardisoinit(pt, Ref{Int32}(-2), iparm)

    # Julia is one-based: iparm[n + 1] corresponds to oneMKL's C iparm[n].
    iparm[1] = 1
    iparm[2] = opt.pardisomkl_order
    iparm[3] = 0
    iparm[6] = 1
    iparm[8] = opt.pardisomkl_max_iterative_refinement_steps
    iparm[10] = 8  # C iparm[9]: perturb tiny pivots at 1e-8 instead of 1e-12.
    iparm[11] = 1  # C iparm[10]: enable scaling.
    iparm[13] = 1  # C iparm[12]: enable symmetric weighted matching.
    iparm[21] = 1  # C iparm[20]: Bunch-Kaufman with automatic refinement.
    iparm[24] = 1
    iparm[25] = 0
    iparm[27] = 1  # C iparm[26]: validate the sparse matrix representation.
    iparm[28] = T == Float64 ? 0 : 1

    MadNLPPardiso.pardisomkl_set_num_threads!(opt.pardisomkl_num_threads)
    MadNLPPardiso.pardisomkl_pardiso(
        pt,
        Ref{Int32}(1),
        Ref{Int32}(1),
        Ref{Int32}(-2),
        Ref{Int32}(11),
        Ref{Int32}(csc.n),
        csc.nzval,
        csc.colptr,
        csc.rowval,
        perm,
        Ref{Int32}(1),
        iparm,
        msglvl,
        T[],
        T[],
        err,
    )
    MadNLPPardiso.pardisomkl_set_num_threads!(MadNLP.blas_num_threads[])
    err.x < 0 && throw(MadNLP.SymbolicException())

    inner = MadNLPPardiso.PardisoMKLSolver{T}(
        pt,
        iparm,
        perm,
        msglvl,
        err,
        csc,
        w,
        MadNLPPardiso.PardisoMKLOptions(
            pardisomkl_num_threads = opt.pardisomkl_num_threads,
            pardisomkl_max_iterative_refinement_steps = opt.pardisomkl_max_iterative_refinement_steps,
            pardisomkl_msglvl = opt.pardisomkl_msglvl,
            pardisomkl_order = opt.pardisomkl_order,
        ),
        logger,
    )
    finalizer(MadNLPPardiso.finalize, inner)
    return RobustPardisoMKLSolver{T}(inner)
end

MadNLP.factorize!(solver::RobustPardisoMKLSolver) = MadNLP.factorize!(solver.inner)
MadNLP.solve_linear_system!(solver::RobustPardisoMKLSolver, rhs::Vector) =
    MadNLP.solve_linear_system!(solver.inner, rhs)
MadNLP.is_inertia(::RobustPardisoMKLSolver) = true
MadNLP.inertia(solver::RobustPardisoMKLSolver) = MadNLP.inertia(solver.inner)
MadNLP.improve!(solver::RobustPardisoMKLSolver) = MadNLP.improve!(solver.inner)
MadNLP.introduce(::RobustPardisoMKLSolver) = "pardiso-mkl-robust"
MadNLP.input_type(::Type{RobustPardisoMKLSolver}) = :csc
MadNLP.default_options(::Type{RobustPardisoMKLSolver}) = RobustPardisoMKLOptions()
MadNLP.is_supported(::Type{RobustPardisoMKLSolver}, ::Type{Float32}) = true
MadNLP.is_supported(::Type{RobustPardisoMKLSolver}, ::Type{Float64}) = true
