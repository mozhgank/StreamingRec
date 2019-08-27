package tudo.streamingrec.algorithms;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import tudo.streamingrec.data.ClickData;
import tudo.streamingrec.data.Transaction;
import tudo.streamingrec.util.Util;

/**
 * An algorithm that extracts and stores sequential patterns from sessions
 * in a tree model to make recommendations.
 * 
 * Code submitted as supplementary material for a under review at RecSys '19 
 * 
 * @author Redacted for blind review
 *
 */
public class SeqPatSimplified extends Algorithm {
	// The support threshold parameter (called sigma in the paper)
	private int supportThreshold = 5;
	// The confidence threshold parameter (called kappa in the paper)
	private double confidenceThreshold = 0.9;
	// A parameter that determines whether stale pattern should be removed 
	// from the model (the concept is called click queue in the paper)
	private boolean recentClickedFiltering = false;
	// If recentClickedFiltering is set to true, the following parameter 
	// determines the click queue/buffer size.
	private int bufferSize = 10000;
	// The following parameter distinguished between the Seq and Seq_p
	// variants of the proposed approach
	private boolean confidenceType = true;
	// Parameter that determines the size of the sliding window 
	// used to limit the size of sub-patterns extracted from each session
	private int windowSize = 10;
	

	// The click queue used to later on remove stale patterns from the model
	private List<List<Transaction>> buffer = new LinkedList<List<Transaction>>();
	// A tree data structure that stores the patterns
	protected SequenceTreeNode patternTree = new SequenceTreeNode();
	// Accumulated support of all nodes in the tree. Used in the strategy Seq_p
	protected int sumSupport = 0;

	/**
	 * Inherited method. See {@link Algorithm#trainInternal(List, List)} for an interface definition.
	 */
	@Override
	protected void trainInternal(java.util.List<tudo.streamingrec.data.Item> items,
			java.util.List<ClickData> clickData) {
		// If recentClickedFiltering is enabled and the incoming data is more than the buffer size,
		// we only need to train on the last part of the training data.
		// The rest of the data is already "outdated" and can be ignored.
		if(recentClickedFiltering && clickData.size()>bufferSize) {
			clickData = clickData.subList(clickData.size()-bufferSize,clickData.size());
		}
		// For all transactions, update the pattern tree
		for (ClickData c : clickData) {
			updateMap(c.session, true);
			// If we only want to learn from the most recent clicks,
			// we need to decrease support counts and remove nodes from the tree
			// whenever clicks become "outdated"
			if (recentClickedFiltering) {
				// Add clicks to the buffer as "new" clicks
				buffer.add(new ArrayList<Transaction>(c.session));
				// Whenever clicks become outdated,
				// start reversing the tree operations, i.e.,
				// reduce support count & remove nodes with 0 support
				if (buffer.size() > bufferSize) {
					List<Transaction> oldClick = buffer.remove(0);
					updateMap(oldClick, false);
				}
			}

		}
	}

	/**
	 * Updates the support values of the pattern tree based on all subpatterns 
	 * extracted from the current session snapshot. If necessary, new nodes are created
	 * if the respective patterns have not yet been encountered in the data.
	 * If the parameter {@code add} is set to false, exactly the opposite operation is 
	 * performed, i.e., the support counts are decreased, to remove stale patterns. 
	 * 
	 * @param session The current session snapshot.
	 * @param add Should the patterns from the current session snapshot be ADDED 
	 * to the tree model or the opposite? In the latter case, a reveseral operation 
	 * is performed, which is used to removed "stale" pattern, i.e., patterns based 
	 * on old clicks from the model.
	 */
	protected void updateMap(List<Transaction> session, boolean add) {
		//========================= 
		// >>> General strategy of this method <<<
		// Extract the necessary sub patterns from the session and add them to
		// the pattern tree.
		// Note: Some patterns have already been added because (due to the
		// incremental learning) the session was already processed before, 
		// but without the most recent click (c.f. Section 3.3 in the paper).
		//========================= 
		
		// Start by removing redundant items from the session, as we are not interested in 
		// recommending items to users that the already saw in the current session.
		session = distinct(session);
		// Apply a sliding window to reduce the maximum pattern size in the session
		// this is mainly to avoid large calculations for the powerset creation,
		// which would otherwise have exponential complexity
		if (session.size() > windowSize) {
			session = session.subList(session.size() - windowSize, session.size());
		}
		
		// Create all subsets of the session that end with the last click 
		// (c.f. Algorithm 1 in the paper)
		List<List<Transaction>> powerSet = powerset(session, true);

		// Iterate over all subsets of the power set of the current session, i.e.,  
		// all possible sub-sequences, and add these click sequences to the tree
		for (List<Transaction> subSet : powerSet) {
			// Start from the root
			SequenceTreeNode currentNode = patternTree;
			// Step down the pattern tree based on the subset from the partial power set.
			// Nodes will be created if necessary along the way. When reaching the final 
			// element of the subset, the support count of that node is increased by 1.
			for (int i = 0; i < subSet.size(); i++) {
				// Go through the subset of the powerset click by click.
				Transaction click = subSet.get(i);
				// Step down the tree based on the current element from the subset.
				if (currentNode.children.containsKey(click.item.id)) {
					currentNode = currentNode.children.get(click.item.id);
				} else {
					//If the necessary node does not exist, create it.
					SequenceTreeNode node = new SequenceTreeNode();
					node.parent = currentNode;
					currentNode.children.put(click.item.id, node);
					currentNode = node;
				}
				// When the last click in the subset is reached,
				// increase the support count of the respective node in the tree.
				if (i == subSet.size() - 1) {
					if (add) {
						// Increase the support count and the sumSupport (which is used in the Seq_p variety)
						currentNode.support++;
						sumSupport++;
					} else {
						// If we are not "adding" to the tree, but instead removing stale patterns, 
						// we should decrease the support count, and, potentially remove nodes with 0 support.
						currentNode.support--;
						sumSupport--;
						// If the support count is 0, we want to remove the node from the tree
						for (int j = subSet.size()-1; j >= 0 ; j--) {
							if(currentNode.support==0  && currentNode.children.isEmpty()) {
								// We only want to remove nodes if they have no children.
								// So, later we may need to go through the parents to search for other nodes that 
								// can be removed as well, because now their children are also zero
								currentNode.parent.children.remove(subSet.get(j).item.id);
								currentNode = currentNode.parent;
							}							
						}
					}
				}
			}
		}
	}

	/**
	 * Inherited method. See {@link Algorithm#recommend(ClickData)} for an interface definition.
	 */
	@Override
	public LongArrayList recommendInternal(ClickData clickData) {
		// Create a score map. Here, for each items, 
		// we keep track of its accumulated confidence scores.
		Map<Long, Double> score = new Long2DoubleOpenHashMap();
		
		// Again, remove redundant items from the session and apply a sliding window.
		List<Transaction> session = distinct(clickData.session);
		if (session.size() > windowSize) {
			session = session.subList(session.size() - windowSize, session.size());
		}

		// Create the COMPLETE power set of the session.
		// We use the whole power set here, as we are not interested in iteratively
		// creating a pattern model, but instead we want to harness all possible subpatterns
		// of the current session for to finding matching pattern in the model.
		List<List<Transaction>> powerSet = powerset(session, false);

		// Iterate over all subsets in the powerset, i.e., all subpatterns.
		for (List<Transaction> subSet : powerSet) {
			if (subSet.isEmpty()) {
				continue; // Ignore the empty subset
			}
			// Step down the pattern tree; start with the root node.
			SequenceTreeNode currentNode = patternTree;

			boolean correctMatch = true;
			// For each item in the subpattern, step down a node in the tree
			for (int i = 0; i < subSet.size(); i++) {
				Transaction click = subSet.get(i);

				//Check whether the tree contain the respective child node
				if (!currentNode.children.containsKey(click.item.id)) {
					// If there is no child node that matches the requested pattern,
					// stop the process and continue with the next subpattern.
					correctMatch = false;
					break;
				}
				// Otherwise, step down.
				currentNode = currentNode.children.get(click.item.id);
			}
			// If, after processing the subpattern completely, we found a matching branch in the tree,
			// we proceed by inspecting the children of the node we arrived at, as these are our 
			// candidate items, i.e., potential pattern continuations for the current user session.
			if (correctMatch) {
				// If we reached the right node, look at the children 
				for (Entry<Long, SequenceTreeNode> child : currentNode.children.entrySet()) {
					// Check whether the node's support count is above the chosen support threshold.
					// If not, do not process it.
					if (child.getValue().support > supportThreshold) {
						if (child.getValue().support == 0 || currentNode.support == 0) {
							continue;
						}
						// Begin to calculate the confidence value 
						// for this candidate item in this specific tree branch						
						double divisor;
						// Depeding on the algorithm variaty (Seq or Seq_p),
						// we use a different divisor in the formula.
						if (confidenceType) {
							divisor = currentNode.support;
						}else {
							divisor = sumSupport;
						}
						// Calculate the confidence score by adding the numerator 
						double childscore = (child.getValue().support * 1d) / divisor;
						// If the confidence score falls under the chosen confidence
						// threshold, do not add the confidence value of this branch 
						// to the current candidate item's accumulated score.
						if (childscore > confidenceThreshold) {
							long key = child.getKey();
							// Add the confidence to the score map for the candidate item.
							if (score.containsKey(key)) {
								score.put(key, (score.get(key) + childscore));
							} else {
								score.put(key, childscore);
							}

						}
					}
				}

			}
		}

		// Sort the accumulated score values and create a recommendation list
		return (LongArrayList) Util.sortByValueAndGetKeys(score, false, new LongArrayList());
	}
	
	/**
	 * Creates a powerset of the given input session snapshot.
	 * If {@code partial} is true, only the partial power set is created.
	 * Refer to Algorithm 1 in the paper for details.
	 * 
	 * @param list The list of items in the current session snapshot
	 * @param partial If true, instead of the complete power set, only the 
	 * partial power set is created. Refer to Algorithm 1 from the paper 
	 * for a details. 
	 * @return all Subsets that end with the last element of the input list
	 */
	public static <T> List<List<T>> powerset(List<T> list, boolean partial) {
		List<List<T>> ps = new ArrayList<List<T>>();
		ps.add(new ArrayList<T>()); // add the empty set

		// If the expected output is a partial powerset,
		// we start based on the session minus the last element.
		// Otherwise, we use the whole session.
		List<T> input = partial?list.subList(0, list.size() - 1):list;
		
		// For every item in the original list
		for (T item : input) {
			List<List<T>> newPs = new ArrayList<List<T>>();

			for (List<T> subset : ps) {
				// copy all of the current powerset's subsets
				newPs.add(subset);

				// plus the subsets appended with the current item
				List<T> newSubset = new ArrayList<T>(subset);
				newSubset.add(item);
				newPs.add(newSubset);
			}

			// powerset is now powerset of list.subList(0, list.indexOf(item)+1)
			ps = newPs;
		}
		
		// If the expected output is a partial powerset,
		// we have to re-attach the last item of the session to
		// each set in the power set.
		if(partial){
			for (List<T> subset : ps) {
				subset.add(list.get(list.size() - 1));
			}
		}		

		return ps;
	}
	
	/**
	 * Removes duplicate items from a click session.
	 * 
	 * @param session The item in the current click session.
	 * @return A click session in which every item appears only once.
	 */
	protected List<Transaction> distinct(List<Transaction> session) {
		Set<Long> idMap = new HashSet<>();
		List<Transaction> dSession = new ArrayList<>();
		for(Transaction t : session){
			if (idMap.add(t.item.id)) {
				// Only add an item to the output list, 
				// if it is not already in the HashSet.
				dSession.add(t);
			}
		}
		return dSession;
	}
	
	/**
	 * Sets the confidence threshold parameter (called kappa in the paper)
	 * @param confidenceThreshold The confidence threshold parameter
	 */
	public void setConfidenceThreshold(double confidenceThreshold) {
		this.confidenceThreshold = confidenceThreshold;
	}

	/**
	 * Sets the support threshold parameter (called sigma in the paper)
	 * @param supportThreshold The support threshold parameter
	 */
	public void setSupportThreshold(int supportThreshold) {
		this.supportThreshold = supportThreshold;
	}

	/**
	 *  Sets the value of the parameter that determines the size of the sliding window
	 *  used to limit the size of sub-patterns extracted from each session
	 * @param windowSize The size of the sliding window
	 */
	public void setWindowSize(int windowSize) {
		this.windowSize = windowSize;
	}

	/**
	 * Sets the recentClickedFiltering parameter. If set to true, the value set via 
	 * {@link #setBufferSize(int)} determines the click queue/buffer size, which is 
	 * responsible for removing stale patterns from the model.
	 * @param recentClickedFiltering The recentClickedFiltering parameter
	 */
	public void setRecentClickedFiltering(boolean recentClickedFiltering) {
		this.recentClickedFiltering = recentClickedFiltering;
	}

	/**
	 * If {@link #setRecentClickedFiltering(boolean)} was set to true,
	 * the value set here determines the size of the click queue, which is used
	 * to remove stale patterns from the model.
	 * @param bufferSize The bufferSize parameter.
	 */
	public void setBufferSize(int bufferSize) {
		this.bufferSize = bufferSize;
	}
	
	/**
	 * Distinguishes between the Seq and Seq<sub>p</sub> varieties of
	 * the algorithm. If set to true, Seq is used; if set to false, Seq<sub>p</sub>.
	 * @param confidenceType The confidenceType parameter.
	 */
	public void setConfidenceType(boolean confidenceType) {
		this.confidenceType = confidenceType;
	}
	
	/**
	 * Represents a tree node in a pattern tree model
	 * (i.e., a database used to store patterns)
	 * 
	 * @author Redacted for blind review
	 *
	 */
	public class SequenceTreeNode {
		// Links to the child nodes. The is the child's item id.
		Map<Long, SequenceTreeNode> children = new Object2ObjectOpenHashMap<Long, SequenceTreeNode>();
		// The support count of this node, i.e., the frequency of this pattern.
		int support = 0;		
		// A backwards link to the parent 
		// (used to remove tree branches, when a child's suppport count becomes 0).
		SequenceTreeNode parent;
	}
}
