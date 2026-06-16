    std::vector<int> valid_choices;
    for (int i = 0; i <= 71; ++i) valid_choices.push_back(i);
    if (priors.pole_at_origin) { for (int i = 0; i < 15; ++i) valid_choices.push_back(60); }
    if (priors.autonomous_x || priors.autonomous_y) { for (int i = 0; i < 15; ++i) valid_choices.push_back(61); }
    if (priors.scale_invariant) { for (int i = 0; i < 15; ++i) valid_choices.push_back(62); }
    int choice = valid_choices[std::uniform_int_distribution<int>(0, valid_choices.size() - 1)(gen)];
    last_special_choice = choice;
    switch (choice) {
        case 4: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_var('x'))));
        case 5: return make_binary(NodeType::DIV, make_var('x'), make_binary(NodeType::ADD, make_erc(1.0), make_unary(NodeType::SQR, make_var('x'))));
        case 6: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))));
        case 7: return make_binary(NodeType::SUB, make_erc(1.0), make_unary(NodeType::GAUSSIAN, make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')))));
        case 8: return make_binary(NodeType::MUL, make_unary(NodeType::TANH, make_var('x')), make_unary(NodeType::TANH, make_var('y')));
        case 9: return make_binary(NodeType::DIV, make_erc(1.0), make_unary(NodeType::COSH, make_binary(NodeType::ADD, make_var('x'), make_var('y'))));
        case 10: return make_binary(NodeType::MUL, make_binary(NodeType::ADD, make_erc(1.0), make_var('x')), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))));
        case 11: return make_binary(NodeType::POW, make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y'))), make_erc(1.0));
        case 12: return make_binary(NodeType::MUL, make_var('t'), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y'))))));
        case 13: return make_binary(NodeType::ADD, make_var('x'), make_binary(NodeType::MUL, make_erc(1.0), make_var('t')));
        case 14: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_var('t'))));
        case 15: return make_binary(NodeType::DIV, make_var('x'), make_binary(NodeType::ADD, make_erc(1.0), make_var('t')));
        case 16: return make_binary(NodeType::MUL, make_var('x'), make_var('y'));
        case 17: return make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')));
        case 18: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))));
        case 19: return make_binary(NodeType::SUB, make_var('x'), make_var('y'));
        case 20: return make_binary(NodeType::DIV, make_var('x'), make_binary(NodeType::ADD, make_erc(1.0), make_var('y')));
        case 21: return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')))));
        case 22: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::EXP, make_var('y')));
        case 23: return make_binary(NodeType::ADD, make_binary(NodeType::MUL, make_erc(1.0), make_var('x')), make_binary(NodeType::MUL, make_erc(1.0), make_var('y')));
        case 24: return make_binary(NodeType::MUL, make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))));
        case 25: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))));
        case 26: return make_binary(NodeType::DIV, make_unary(NodeType::SQR, make_var('x')), make_binary(NodeType::ADD, make_erc(1.0), make_unary(NodeType::SQR, make_var('y'))));
        case 27: return make_binary(NodeType::ADD, make_var('x'), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))));
        case 28: return make_binary(NodeType::MUL, make_unary(NodeType::EXP, make_var('x')), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))));
        case 29: return make_binary(NodeType::SUB, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')));
        case 30: return make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::MUL, make_var('x'), make_var('y')));
        case 31: return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')))));
        case 32: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_binary(NodeType::ADD, make_var('x'), make_var('y')))), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_var('t'))));
        case 33: return make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_binary(NodeType::MUL, make_var('y'), make_var('t')));
        case 34: return make_binary(NodeType::DIV, make_var('x'), make_binary(NodeType::ADD, make_erc(1.0), make_unary(NodeType::SQR, make_var('t'))));
        case 35: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), make_binary(NodeType::ADD, make_var('y'), make_var('t')))));
        case 36: return make_binary(NodeType::SUB, make_var('x'), make_binary(NodeType::MUL, make_var('y'), make_var('t')));
        case 37: return make_binary(NodeType::ADD, make_unary(NodeType::EXP, make_var('x')), make_binary(NodeType::MUL, make_erc(1.0), make_var('t')));
        case 38: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), make_var('t'))));
        case 39: return make_binary(NodeType::DIV, make_binary(NodeType::ADD, make_var('x'), make_var('y')), make_binary(NodeType::ADD, make_erc(1.0), make_var('t')));
        case 40: return make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('t')));
        case 41: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_var('t'))));
        case 42: return make_binary(NodeType::ADD, make_var('x'), make_binary(NodeType::ADD, make_var('y'), make_var('t')));
        case 43: return make_binary(NodeType::MUL, make_var('x'), make_binary(NodeType::MUL, make_var('y'), make_var('t')));
        case 44: return make_binary(NodeType::SUB, make_binary(NodeType::ADD, make_var('x'), make_var('y')), make_var('t'));
        case 45: return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::ADD, make_var('x'), make_binary(NodeType::ADD, make_var('y'), make_var('t')))));
        case 46: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))), make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('t')))));
        case 47: return make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_binary(NodeType::SUB, make_unary(NodeType::SQR, make_var('y')), make_unary(NodeType::SQR, make_var('t'))));
        case 48: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_var('y'), make_var('t'))));
        case 49: return make_binary(NodeType::ADD, make_binary(NodeType::MUL, make_erc(1.0), make_var('x')), make_binary(NodeType::MUL, make_erc(1.0), make_var('t')));
        case 50: return make_binary(NodeType::DIV, make_var('x'), make_binary(NodeType::ADD, make_var('y'), make_var('t')));
        case 51: return make_binary(NodeType::MUL, make_unary(NodeType::TANH, make_var('x')), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_var('t'))));
        case 52: return make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_binary(NodeType::MUL, make_erc(1.0), make_var('t')));
        case 53: {
            auto nx = make_binary(NodeType::MUL, make_const_pi(), make_var('x'));
            auto ny = make_binary(NodeType::MUL, make_const_pi(), make_var('y'));
            auto nt = make_binary(NodeType::MUL, make_erc(1.0), make_var('t'));
            return std::make_unique<SeriesNode>(3, make_binary(NodeType::MUL, make_binary(NodeType::MUL, make_unary(NodeType::SIN, std::move(nx)), make_unary(NodeType::SIN, std::move(ny))), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), std::move(nt)))));
        }
        case 54: return make_binary(NodeType::ADD, make_binary(NodeType::MUL, make_erc(1.0), make_var('x')), make_binary(NodeType::MUL, make_erc(1.0), make_var('y')));
        case 55: return make_binary(NodeType::ADD, make_binary(NodeType::ADD, make_binary(NodeType::MUL, make_erc(1.0), make_var('x')), make_binary(NodeType::MUL, make_erc(1.0), make_var('y'))), make_binary(NodeType::MUL, make_erc(1.0), make_var('t')));
        case 56: return make_binary(NodeType::MUL, make_erc(1.0), make_binary(NodeType::MUL, make_var('x'), make_var('y')));
        case 57: return make_binary(NodeType::MUL, make_erc(1.0), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))));
        case 58: return make_binary(NodeType::MUL, make_var('x'), make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_unary(NodeType::SQR, make_var('x')))));
        case 59: return make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::MUL, make_erc(1.0), make_unary(NodeType::SQR, make_var('x'))));
        case 60: return skeleton_rational(depth, gen, prob);
        case 61: return skeleton_spectral(depth, gen, prob);
        case 62: return skeleton_frobenius(depth, gen, prob);
        case 63: { // Vórtice de Lamb-Oseen: (1 - exp(-r^2)) / r^2
            auto r2_a = make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')));
            auto r2_b = make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')));
            auto exp_term = make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), std::move(r2_a)));
            auto sub_term = make_binary(NodeType::SUB, make_erc(1.0), std::move(exp_term));
            return make_binary(NodeType::DIV, std::move(sub_term), std::move(r2_b));
        }
        case 64: return make_binary(NodeType::SUB, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y'))); 
        case 65: return make_binary(NodeType::MUL, make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_var('t'))), make_unary(NodeType::COS, make_binary(NodeType::MUL, make_const_pi(), make_var('x')))); 
        case 66: return make_binary(NodeType::DIV, make_binary(NodeType::ADD, make_erc(1.0), make_var('x')), make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::ADD, make_var('x'), make_unary(NodeType::SQR, make_var('x'))))); 
        case 67: return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::POW, make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_erc(1.0)), make_erc(1.0))); 
        case 68: { auto fx = make_binary(NodeType::ADD, make_binary(NodeType::MUL, make_erc(1.0), make_var('x')), make_erc(1.0));
            auto gy = make_binary(NodeType::ADD, make_binary(NodeType::MUL, make_erc(1.0), make_var('y')), make_erc(1.0));
            return make_binary(NodeType::MUL, std::move(fx), std::move(gy)); }
        case 69: { auto ex = make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(1.0), make_var('x')));
            auto sy = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y')));
            return make_binary(NodeType::MUL, std::move(ex), std::move(sy)); }
        case 0: return make_unary(NodeType::TANH, make_binary(NodeType::SUB, make_var('x'), make_binary(NodeType::MUL, make_erc(1.0), make_var('t'))));
        case 1: return make_binary(NodeType::MUL, make_var('x'), make_binary(NodeType::ADD, make_const_pi(), make_var('t')));
        case 2: return make_binary(NodeType::MUL, make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_binary(NodeType::SUB, make_var('x'), make_var('t')))), make_unary(NodeType::GAUSSIAN, make_var('x')));
        case 3: return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_var('t'))));
        case 70: { /* Kovasznay Skeleton (Agnostic + PI) */ 
            auto ex = make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(1.0), make_var('x'))); 
            auto sy = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_binary(NodeType::MUL, make_erc(2.0), make_const_pi()), make_var('y'))); 
            auto term = make_binary(NodeType::MUL, make_erc(1.0), make_binary(NodeType::MUL, std::move(ex), std::move(sy)));
            return make_binary(NodeType::SUB, make_var('y'), std::move(term)); 
        }
        case 71: { /* Unsteady Decay Skeleton (Agnostic + PI) */ 
            auto sx = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x'))); 
            auto sy = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y'))); 
            auto et = make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-1.0), make_var('t'))); 
            return make_binary(NodeType::MUL, std::move(sx), make_binary(NodeType::MUL, std::move(sy), std::move(et))); 
        }
        default: return get_exact_solution_tree(prob);
    }
}
