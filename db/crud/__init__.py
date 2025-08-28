# db/crud/__init__.py

# Import user functions from users.py
# Import user functions from users.py
from .users import (
    get_user_by_email,
    create_user,
    get_user_by_id  # Add this
)

# Import portfolio and risk functions from portfolios.py
from .portfolios import (
    get_user_portfolios,
    get_latest_risk_snapshot,
    get_threshold_breaches,
    get_portfolio_by_id  # Add this
)

# Import debate functions from debates.py
from .debates import (
    create_debate, get_debate, get_user_debates, update_debate_status,
    update_debate_results, delete_debate,

    # Participant operations
    add_debate_participant, get_debate_participants, update_participant_activity,
    update_participant_final_position, bulk_create_participants, bulk_update_participant_metrics,

    # Message operations
    create_debate_message, get_debate_messages, update_message_analysis,
    get_message_thread, bulk_analyze_messages,

    # Consensus operations
    create_consensus_item, add_consensus_vote, get_debate_consensus_items,
    update_consensus_item_status,

    # Analytics operations
    create_debate_analytics, get_debate_analytics, get_user_debate_analytics,

    # Template operations
    create_debate_template, get_debate_templates, update_template_usage_stats,

    # Performance operations
    create_agent_performance_record, get_agent_performance_history, get_top_performing_agents,

    # Advanced operations
    search_debates, get_debate_statistics, get_agent_collaboration_matrix,
    cleanup_old_debates, export_debate_data, get_debate_performance_report,

    # Utility functions
    validate_debate_state_transition, calculate_debate_health_score,

    # Enums and filters
    DebateQueryFilter
)

# You can optionally define __all__ to control what `from db.crud import *` imports
__all__ = [
    # users.py
    'get_user_by_email', 'create_user',
    # portfolios.py
    'get_user_portfolios', 'get_latest_risk_snapshot', 'get_threshold_breaches',
    # debates.py
    'create_debate', 'get_debate', 'get_user_debates', 'update_debate_status',
    'update_debate_results', 'delete_debate', 'add_debate_participant',
    'get_debate_participants', 'update_participant_activity', 'update_participant_final_position',
    'bulk_create_participants', 'bulk_update_participant_metrics', 'create_debate_message',
    'get_debate_messages', 'update_message_analysis', 'get_message_thread',
    'bulk_analyze_messages', 'create_consensus_item', 'add_consensus_vote',
    'get_debate_consensus_items', 'update_consensus_item_status', 'create_debate_analytics',
    'get_debate_analytics', 'get_user_debate_analytics', 'create_debate_template',
    'get_debate_templates', 'update_template_usage_stats', 'create_agent_performance_record',
    'get_agent_performance_history', 'get_top_performing_agents', 'search_debates',
    'get_debate_statistics', 'get_agent_collaboration_matrix', 'cleanup_old_debates',
    'export_debate_data', 'get_debate_performance_report', 'validate_debate_state_transition',
    'calculate_debate_health_score', 'DebateQueryFilter'
]