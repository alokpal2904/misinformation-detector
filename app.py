import streamlit as st
import pandas as pd
from datetime import datetime
import time
from detector import MisinformationDetector
import threading

# Initialize the detector
@st.cache_resource
def get_detector():
    return MisinformationDetector()

def main():
    st.set_page_config(
        page_title="Real-Time Misinformation Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Real-Time Misinformation Detector")
    st.markdown("Monitoring Reddit for trending misinformation")
    
    detector = get_detector()
    
    # Start the detector in a background thread if not already running
    if not hasattr(st.session_state, 'detector_started'):
        detector_thread = threading.Thread(target=detector.run)
        detector_thread.daemon = True
        detector_thread.start()
        st.session_state.detector_started = True
    
    # Create placeholder for dynamic content
    claims_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Add refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()
    
    while True:
        with claims_placeholder.container():
            st.header("Recently Verified Claims")
            
            recent_claims = detector.get_recent_claims(10)
            
            if recent_claims:
                # Convert to DataFrame for better display
                df_data = []
                for claim in recent_claims:
                    # Add color coding based on verdict
                    verdict_color = {
                        'true': '🟢',
                        'false': '🔴', 
                        'misleading': '🟡',
                        'unverifiable': '⚪'
                    }.get(claim['verdict'], '⚪')
                    
                    df_data.append({
                        'Claim': claim['post']['title'][:100] + "..." if len(claim['post']['title']) > 100 else claim['post']['title'],
                        'Verdict': f"{verdict_color} {claim['verdict'].upper()}",
                        'Confidence': f"{claim.get('confidence', 0)}%",
                        'Subreddit': claim['post']['subreddit'],
                        'Timestamp': datetime.fromisoformat(claim['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, width='stretch')
                
                # Show details for selected claim
                if len(df_data) > 0:
                    selected_idx = st.selectbox(
                        "Select claim to view details", 
                        range(len(df_data)), 
                        format_func=lambda x: f"{df_data[x]['Claim'][:50]}... ({df_data[x]['Verdict']})"
                    )
                    
                    selected_claim = recent_claims[selected_idx]
                    with st.expander("Claim Details", expanded=True):
                        st.write(f"**Full Claim:** {selected_claim['post']['title']}")
                        
                        # Color code the verdict
                        verdict_color = {
                            'true': 'green',
                            'false': 'red',
                            'misleading': 'orange',
                            'unverifiable': 'gray'
                        }.get(selected_claim['verdict'], 'gray')
                        
                        st.markdown(f"**Verdict:** <span style='color:{verdict_color};font-weight:bold'>{selected_claim['verdict'].upper()}</span> ({selected_claim.get('confidence', 0)}% confidence)", 
                                   unsafe_allow_html=True)
                        
                        st.write(f"**Explanation:** {selected_claim.get('explanation', 'No explanation provided')}")
                        st.write(f"**Subreddit:** r/{selected_claim['post']['subreddit']}")
                        st.write(f"**URL:** {selected_claim['post']['url']}")
                        
                        sources = selected_claim.get('sources', [])
                        if sources:
                            st.write("**Sources:**")
                            for source in sources:
                                if isinstance(source, str) and source.startswith('http'):
                                    st.write(f"- [{source}]({source})")
                                else:
                                    st.write(f"- {source}")
                        else:
                            st.write("**Sources:** No sources provided")
            else:
                st.info("No claims verified yet. Monitoring in progress...")
                st.write("The system is currently scanning Reddit for trending topics and will display results here once claims are verified.")
        
        with stats_placeholder.container():
            st.header("📊 Statistics")
            
            all_claims = detector.verified_claims
            false_claims = sum(1 for c in all_claims if c['verdict'] in ['false', 'misleading'])
            true_claims = sum(1 for c in all_claims if c['verdict'] == 'true')
            unverifiable = sum(1 for c in all_claims if c['verdict'] == 'unverifiable')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Claims Checked", len(all_claims))
            with col2:
                st.metric("Misinformation Detected", false_claims, 
                         delta=f"{false_claims/len(all_claims)*100:.1f}%" if all_claims else 0)
            with col3:
                st.metric("Accurate Claims", true_claims,
                         delta=f"{true_claims/len(all_claims)*100:.1f}%" if all_claims else 0)
            with col4:
                st.metric("Unverifiable Claims", unverifiable,
                         delta=f"{unverifiable/len(all_claims)*100:.1f}%" if all_claims else 0)
            
            # Add a small status indicator
            st.write("---")
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                if detector.reddit_scraper.last_check:
                    last_check_time = datetime.fromisoformat(str(detector.reddit_scraper.last_check)) if isinstance(detector.reddit_scraper.last_check, str) else detector.reddit_scraper.last_check
                    st.write(f"**Last Check:** {last_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Queue Size:** {detector.claims_queue.qsize()}")
            with status_col2:
                if detector.verified_claims:
                    latest_claim = detector.verified_claims[-1]
                    latest_time = datetime.fromisoformat(latest_claim['timestamp'])
                    st.write(f"**Latest Result:** {latest_time.strftime('%H:%M:%S')}")
                st.write(f"**Active Threads:** {threading.active_count()}")
        
        # Update every 10 seconds
        time.sleep(10)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from datetime import datetime
import time
from detector import MisinformationDetector
import threading
import hashlib

# Initialize the detector
@st.cache_resource
def get_detector():
    return MisinformationDetector()

def main():
    st.set_page_config(
        page_title="Real-Time Misinformation Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Real-Time Misinformation Detector")
    st.markdown("Monitoring Reddit for trending misinformation - Processing batches every 2 minutes")
    
    detector = get_detector()
    
    # Start the detector in a background thread if not already running
    if not hasattr(st.session_state, 'detector_started'):
        detector_thread = threading.Thread(target=detector.run)
        detector_thread.daemon = True
        detector_thread.start()
        st.session_state.detector_started = True
    
    # Create placeholder for dynamic content
    claims_placeholder = st.empty()
    stats_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Add refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()
    
    # Initialize session state for selectbox key
    if 'selectbox_key' not in st.session_state:
        st.session_state.selectbox_key = 0
    
    while True:
        with status_placeholder.container():
            st.header("⏰ Status")
            next_batch_time = detector.get_next_batch_time()
            time_until_next = next_batch_time - datetime.now()
            minutes, seconds = divmod(time_until_next.total_seconds(), 60)
            
            st.write(f"**Next batch processing:** {next_batch_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Time until next batch:** {int(minutes)} minutes {int(seconds)} seconds")
            st.write(f"**Claims in queue:** {detector.claims_queue.qsize()}")
            st.write(f"**Total claims processed:** {len(detector.verified_claims)}")
        
        with claims_placeholder.container():
            st.header("Recently Verified Claims")
            
            recent_claims = detector.get_recent_claims(10)
            
            if recent_claims:
                # Convert to DataFrame for better display
                df_data = []
                for claim in recent_claims:
                    # Add color coding based on verdict
                    verdict_color = {
                        'true': '🟢',
                        'false': '🔴', 
                        'misleading': '🟡',
                        'unverifiable': '⚪'
                    }.get(claim['verdict'], '⚪')
                    
                    df_data.append({
                        'Claim': claim['post']['title'][:100] + "..." if len(claim['post']['title']) > 100 else claim['post']['title'],
                        'Verdict': f"{verdict_color} {claim['verdict'].upper()}",
                        'Confidence': f"{claim.get('confidence', 0)}%",
                        'Subreddit': claim['post']['subreddit'],
                        'Timestamp': datetime.fromisoformat(claim['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, width='stretch')
                
                # Show details for selected claim
                if len(df_data) > 0:
                    # Create a unique key for the selectbox
                    key_suffix = hashlib.md5(str([c['post']['id'] for c in recent_claims]).encode()).hexdigest()
                    
                    selected_idx = st.selectbox(
                        "Select claim to view details", 
                        range(len(df_data)), 
                        format_func=lambda x: f"{df_data[x]['Claim'][:50]}... ({df_data[x]['Verdict']})",
                        key=f"claim_select_{key_suffix}"
                    )
                    
                    selected_claim = recent_claims[selected_idx]
                    with st.expander("Claim Details", expanded=True):
                        st.write(f"**Full Claim:** {selected_claim['post']['title']}")
                        
                        # Color code the verdict
                        verdict_color = {
                            'true': 'green',
                            'false': 'red',
                            'misleading': 'orange',
                            'unverifiable': 'gray'
                        }.get(selected_claim['verdict'], 'gray')
                        
                        st.markdown(f"**Verdict:** <span style='color:{verdict_color};font-weight:bold'>{selected_claim['verdict'].upper()}</span> ({selected_claim.get('confidence', 0)}% confidence)", 
                                   unsafe_allow_html=True)
                        
                        st.write(f"**Explanation:** {selected_claim.get('explanation', 'No explanation provided')}")
                        st.write(f"**Subreddit:** r/{selected_claim['post']['subreddit']}")
                        st.write(f"**URL:** {selected_claim['post']['url']}")
                        
                        sources = selected_claim.get('sources', [])
                        if sources:
                            st.write("**Sources:**")
                            for source in sources:
                                if isinstance(source, str) and source.startswith('http'):
                                    st.write(f"- [{source}]({source})")
                                else:
                                    st.write(f"- {source}")
                        else:
                            st.write("**Sources:** No sources provided")
            else:
                st.info("No claims verified yet. Monitoring in progress...")
                st.write("The system is currently scanning Reddit for trending topics and will display results here once claims are verified.")
        
        with stats_placeholder.container():
            st.header("📊 Statistics")
            
            all_claims = detector.verified_claims
            false_claims = sum(1 for c in all_claims if c['verdict'] in ['false', 'misleading'])
            true_claims = sum(1 for c in all_claims if c['verdict'] == 'true')
            unverifiable = sum(1 for c in all_claims if c['verdict'] == 'unverifiable')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Claims Checked", len(all_claims))
            with col2:
                st.metric("Misinformation Detected", false_claims, 
                         delta=f"{false_claims/len(all_claims)*100:.1f}%" if all_claims else 0)
            with col3:
                st.metric("Accurate Claims", true_claims,
                         delta=f"{true_claims/len(all_claims)*100:.1f}%" if all_claims else 0)
            with col4:
                st.metric("Unverifiable Claims", unverifiable,
                         delta=f"{unverifiable/len(all_claims)*100:.1f}%" if all_claims else 0)
            
            # Add a small status indicator
            st.write("---")
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                if detector.reddit_scraper.last_check:
                    last_check_time = datetime.fromisoformat(str(detector.reddit_scraper.last_check)) if isinstance(detector.reddit_scraper.last_check, str) else detector.reddit_scraper.last_check
                    st.write(f"**Last Reddit Check:** {last_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Queue Size:** {detector.claims_queue.qsize()}")
            with status_col2:
                if detector.verified_claims:
                    latest_claim = detector.verified_claims[-1]
                    latest_time = datetime.fromisoformat(latest_claim['timestamp'])
                    st.write(f"**Latest Result:** {latest_time.strftime('%H:%M:%S')}")
                st.write(f"**Active Threads:** {threading.active_count()}")
        
        # Update every 10 seconds
        time.sleep(10)

if __name__ == "__main__":
    main()

