/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <ctime>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Utils.h"

using namespace Utils;

UCTNode::UCTNode(int vertex, float score) : m_move(vertex), m_score(score) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

SMP::Mutex& UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_children(std::atomic<int>& nodecount,
                              GameState& state,
                              float& eval) {
    // check whether somebody beat us to it (atomic)
    if (has_children()) {
        return false;
    }
    // acquire the lock
    LOCK(get_mutex(), lock);
    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }
    // check whether somebody beat us to it (after taking the lock)
    if (has_children()) {
        return false;
    }
    // Someone else is running the expansion
    if (m_is_expanding) {
        return false;
    }
    // We'll be the one queueing this node for expansion, stop others
    m_is_expanding = true;
    lock.unlock();

    const auto raw_netlist = Network::get_scored_moves(
        &state, Network::Ensemble::RANDOM_ROTATION);

    // DCNN returns winrate as side to move
    m_net_eval = raw_netlist.second;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move()) {
        m_net_eval = 1.0f - m_net_eval;
    }
    eval = m_net_eval;

    std::vector<Network::scored_node> nodelist;

    auto legal_sum = 0.0f;
    for (const auto& node : raw_netlist.first) {
        auto vertex = node.second;
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(node);
            legal_sum += node.first;
        }
    }

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    } else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    link_nodelist(nodecount, nodelist);
    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::scored_node>& nodelist) {
    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
	std::stable_sort(rbegin(nodelist), rend(nodelist));

    LOCK(get_mutex(), lock);

    m_children.reserve(nodelist.size());
    for (const auto& node : nodelist) {
        m_children.emplace_back(
            std::make_unique<UCTNode>(node.second, node.first)
        );
    }

    nodecount += m_children.size();
    m_has_children = true;
}

const std::vector<UCTNode::node_ptr_t>& UCTNode::get_children() const {
    return m_children;
}


int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::update(float eval) {
    m_visits++;
    accumulate_eval(eval);
}

bool UCTNode::has_children() const {
    return m_has_children;
}

float UCTNode::get_score() const {
    return m_score;
}

void UCTNode::set_score(float score) {
    m_score = score;
}

int UCTNode::get_visits() const {
    return m_visits;
}

float UCTNode::get_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    auto virtual_loss = int{m_virtual_loss};
    auto visits = get_visits() + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
    }
    auto score = static_cast<float>(blackeval / (double)visits);
    if (tomove == FastBoard::WHITE) {
        score = 1.0f - score;
    }
    return score;
}

float UCTNode::get_net_eval(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_net_eval;
    }
    return m_net_eval;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, (double)eval);
}

UCTNode* UCTNode::uct_select_child(int color, bool is_root, int movenum, bool pondering_now) {
    UCTNode* best = nullptr;
    //auto best_value = -1000.0; // replaced with next line:
	auto best_value = std::numeric_limits<double>::lowest();
	auto best_winrate = std::numeric_limits<double>::lowest();
	auto best_puct = std::numeric_limits<double>::lowest();

    LOCK(get_mutex(), lock);

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};
    for (const auto& child : m_children) {
        if (child->valid()) {
            parentvisits += child->get_visits();
            if (child->get_visits() > 0) {
                total_visited_policy += child->get_score();
            }
        }
    }

    auto numerator = std::sqrt((double)parentvisits);
    //auto fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy); // replaced with next 7 lines:
	auto fpu_reduction = 0.0f;
	    // Lower the expected eval for moves that are likely not the best.
		    // Do not do this if we have introduced noise at this node exactly
		    // to explore more.
		if (!is_root || !cfg_noise) {
		fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy);
		}

    // Estimated eval for unknown nodes = original parent NN eval - reduction
    auto fpu_eval = get_net_eval(color) - fpu_reduction;

	for (const auto& child : m_children) {
		if (!child->active()) {
			continue;
		}

		float winrate = fpu_eval;

		if (child->get_visits() > 0) {
			winrate = child->get_eval(color);
		}
		auto psa = child->get_score();
		auto denom = 1.0 + child->get_visits();
		auto puct = cfg_puct * psa * (numerator / denom);
		//if (puct <= 0.000001) {
		//	puct = 0.000001;
		//}
		//if (puct >= 0.01) {
		//	puct = 0.01;
		//}
		//if (best_puct <= 0.000001) {
		//	best_puct = 0.000001;
		//}
		auto value = winrate + puct;
		if (movenum < 100) {
			value = (winrate) + (0.01 * std::sqrt(puct));
		}


	//TODO: This is the "coin flip" logic for randomized fuseki. I've disabled/commented it out for now, but it works straight away.
	/*
		auto value = winrate + puct; //NOTE: This line is duplicated above. Probably I shouldn't do that.
		//myprintf("winrate %5.2f -> puct %5.2f\n", winrate, puct);
		if (movenum < 20) {
			int flip_coin = rand() % 80;
			if ((movenum / (1 + flip_coin)) <= 1) {
				value = (winrate - (1.0 * puct)) + (1.0 * (2 * puct * (movenum / 19)));
			}
		}

	*/

		//TODO: THE FOLLOWING SECTION IS ALSO THE FUSEKI STUFF. JUST REMOVE THE ENTIRE COMMENT BLOCK TO REENABLE IT.

		/*
		assert(value > std::numeric_limits<double>::lowest());
		assert(winrate > std::numeric_limits<double>::lowest());
		assert(puct > std::numeric_limits<double>::lowest());
		if (value >= (0.9 * best_value) && puct <= best_puct && movenum < -1) {
			if (value > best_value) {
				best_value = value;
				best_puct = puct;
				best_winrate = winrate;
			}
			best = child.get();
			assert(best != nullptr);
			return best;
		}
		else if (winrate >= (0.9 * winrate) && movenum < -1) {
			best_value = value;
			best_winrate = winrate;
			best_puct = puct;
			best = child.get();
			assert(best != nullptr);
			return best;
		}
		*/

		//assert(value > -1000.0); // replaced with next line:

		//if (child->get_visits() == 0) {	// Get at least more than 0 visits
		//	//best_value = value;
		//	best = child.get();
		//	assert(best != nullptr);
		//	return best;
		//}
		//if (child->get_visits() >= 10 && child->get_visits() < 100) {	// Else if more than 10 visits, have at least 100 visits
		//	//best_value = value;
		//	best = child.get();
		//	assert(best != nullptr);
		//	return best;
		//}

		if (value > best_value) {
			best_value = value;
			best = child.get();
		}
    }

    assert(best != nullptr);
    return best;
}

class NodeComp : public std::binary_function<UCTNode::node_ptr_t&,
                                             UCTNode::node_ptr_t&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNode::node_ptr_t& a,
                    const UCTNode::node_ptr_t& b) {

		////////////Next Line:  FORCE RETURN PRIOR SCORE ONLY
		/////New next line:  If visits are greater than 100, then sort on SCORE
		// EVEN NEWER LINE: First sort by pure visits. Then sort by score IF visits are more than 100.
		
		a->get_visits() < b->get_visits();
		if (a->get_visits() >= 100 && b->get_visits() >= 100) {
		return a->get_eval(m_color) < b->get_eval(m_color);
		} // test test2 test3 test4 test5
        // if visits are not same, sort on visits
        if (a->get_visits() != b->get_visits()) {
            return a->get_visits() < b->get_visits();
        }

        // neither has visits, sort on prior score
        if (a->get_visits() == 0) {
            return a->get_score() < b->get_score();
        }

        // both have same non-zero number of visits
        return a->get_eval(m_color) < b->get_eval(m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_children(int color) {
	LOCK(get_mutex(), lock);
	std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color));
}

void UCTNode::sort_children_reverse(int color) {
	LOCK(get_mutex(), lock);
	std::stable_sort(begin(m_children), end(m_children), NodeComp(color));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    LOCK(get_mutex(), lock);
    assert(!m_children.empty());

    return *(std::max_element(begin(m_children), end(m_children),
                              NodeComp(color))->get());
}

size_t UCTNode::count_nodes() const {
    auto nodecount = size_t{0};
    if (m_has_children) {
        nodecount += m_children.size();
        for (auto& child : m_children) {
            nodecount += child->count_nodes();
        }
    }
    return nodecount;
}

void UCTNode::invalidate() {
    m_status = INVALID;
}

void UCTNode::set_active(const bool active) {
    if (valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

bool UCTNode::valid() const {
    return m_status != INVALID;
}

bool UCTNode::active() const {
    return m_status == ACTIVE;
}
