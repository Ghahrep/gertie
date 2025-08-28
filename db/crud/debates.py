# db/crud/debates.py
"""
CRUD Operations for Multi-Agent Debate System
===========================================
Database operations for creating, reading, updating, and deleting
debate records with optimized queries and analytics.
"""

from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import and_, or_, func, desc, asc, text
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid
from enum import Enum

from db.models import (
    Debate, DebateParticipant, DebateMessage, ConsensusItem, ConsensusVote,
    DebateAnalytics, DebateTemplate, AgentPerformanceHistory,
    DebateStatus, DebateStage, MessageType, ConsensusType
)
from db.models import calculate_debate_quality_score

class DebateQueryFilter(Enum):
    """Common debate query filters"""
    ACTIVE = "active"
    COMPLETED = "completed"
    USER_DEBATES = "user_debates"
    RECENT = "recent"
    HIGH_QUALITY = "high_quality"

# ==========================================
# DEBATE CRUD OPERATIONS
# ==========================================

def create_debate(
    db: Session,
    user_id: int,
    query: str,
    portfolio_id: Optional[str] = None,
    description: Optional[str] = None,
    max_rounds: int = 3,
    urgency_level: str = "medium",
    max_duration_seconds: int = 900,
    require_unanimous_consensus: bool = False,
    include_minority_report: bool = True
) -> Debate:
    """Create a new debate"""
    
    debate = Debate(
        user_id=user_id,
        portfolio_id=uuid.UUID(portfolio_id) if portfolio_id else None,
        query=query,
        description=description,
        max_rounds=max_rounds,
        urgency_level=urgency_level,
        max_duration_seconds=max_duration_seconds,
        require_unanimous_consensus=require_unanimous_consensus,
        include_minority_report=include_minority_report,
        status=DebateStatus.PENDING,
        current_stage=DebateStage.POSITION_FORMATION,
        current_round=1,
        created_at=datetime.utcnow(),
        last_activity_at=datetime.utcnow()
    )
    
    db.add(debate)
    db.commit()
    db.refresh(debate)
    
    return debate

def get_debate(db: Session, debate_id: str, include_messages: bool = False, 
               include_analytics: bool = False) -> Optional[Debate]:
    """Get debate by ID with optional related data"""
    
    query = db.query(Debate).filter(Debate.id == uuid.UUID(debate_id))
    
    # Add eager loading for related data
    load_options = [joinedload(Debate.participants)]
    
    if include_messages:
        load_options.append(selectinload(Debate.messages))
    
    if include_analytics:
        load_options.append(joinedload(Debate.analytics))
    
    query = query.options(*load_options)
    
    return query.first()

def get_user_debates(
    db: Session,
    user_id: int,
    filter_type: Optional[DebateQueryFilter] = None,
    limit: int = 50,
    offset: int = 0,
    include_completed: bool = True
) -> List[Debate]:
    """Get debates for a user with filtering"""
    
    query = db.query(Debate).filter(Debate.user_id == user_id)
    
    # Apply filters
    if filter_type == DebateQueryFilter.ACTIVE:
        query = query.filter(Debate.status.in_([DebateStatus.PENDING, DebateStatus.ACTIVE]))
    elif filter_type == DebateQueryFilter.COMPLETED:
        query = query.filter(Debate.status == DebateStatus.COMPLETED)
    elif filter_type == DebateQueryFilter.RECENT:
        week_ago = datetime.utcnow() - timedelta(days=7)
        query = query.filter(Debate.created_at >= week_ago)
    elif filter_type == DebateQueryFilter.HIGH_QUALITY:
        query = query.filter(Debate.confidence_score >= 0.8)
    
    if not include_completed:
        query = query.filter(Debate.status != DebateStatus.COMPLETED)
    
    # Order by most recent activity
    query = query.order_by(desc(Debate.last_activity_at))
    
    # Add pagination
    query = query.offset(offset).limit(limit)
    
    # Eager load basic related data
    query = query.options(
        joinedload(Debate.participants),
        joinedload(Debate.analytics)
    )
    
    return query.all()

def update_debate_status(
    db: Session,
    debate_id: str,
    status: DebateStatus,
    current_stage: Optional[DebateStage] = None,
    current_round: Optional[int] = None
) -> Optional[Debate]:
    """Update debate status and stage"""
    
    debate = db.query(Debate).filter(Debate.id == uuid.UUID(debate_id)).first()
    
    if not debate:
        return None
    
    debate.status = status
    debate.last_activity_at = datetime.utcnow()
    
    if current_stage:
        debate.current_stage = current_stage
    
    if current_round:
        debate.current_round = current_round
    
    # Set timing fields based on status
    if status == DebateStatus.ACTIVE and not debate.started_at:
        debate.started_at = datetime.utcnow()
    elif status == DebateStatus.COMPLETED:
        debate.completed_at = datetime.utcnow()
        if debate.started_at:
            debate.duration_seconds = (debate.completed_at - debate.started_at).total_seconds()
    
    db.commit()
    db.refresh(debate)
    
    return debate

def update_debate_results(
    db: Session,
    debate_id: str,
    final_recommendation: Dict[str, Any],
    consensus_type: ConsensusType,
    confidence_score: float,
    minority_opinions: Optional[List[Dict]] = None,
    implementation_guidance: Optional[Dict] = None
) -> Optional[Debate]:
    """Update debate with final results"""
    
    debate = db.query(Debate).filter(Debate.id == uuid.UUID(debate_id)).first()
    
    if not debate:
        return None
    
    debate.final_recommendation = final_recommendation
    debate.consensus_type = consensus_type
    debate.confidence_score = confidence_score
    debate.minority_opinions = minority_opinions or []
    debate.implementation_guidance = implementation_guidance or {}
    debate.last_activity_at = datetime.utcnow()
    
    db.commit()
    db.refresh(debate)
    
    return debate

def delete_debate(db: Session, debate_id: str, user_id: int) -> bool:
    """Delete debate (only by owner)"""
    
    debate = db.query(Debate).filter(
        and_(Debate.id == uuid.UUID(debate_id), Debate.user_id == user_id)
    ).first()
    
    if not debate:
        return False
    
    db.delete(debate)
    db.commit()
    
    return True

# ==========================================
# DEBATE PARTICIPANT CRUD OPERATIONS
# ==========================================

def add_debate_participant(
    db: Session,
    debate_id: str,
    agent_id: str,
    agent_name: str,
    agent_type: str,
    agent_specialization: str,
    role: str = "participant"
) -> DebateParticipant:
    """Add participant to debate"""
    
    participant = DebateParticipant(
        debate_id=uuid.UUID(debate_id),
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type,
        agent_specialization=agent_specialization,
        role=role,
        joined_at=datetime.utcnow(),
        last_active_at=datetime.utcnow()
    )
    
    db.add(participant)
    
    # Update participant count on debate
    debate = db.query(Debate).filter(Debate.id == uuid.UUID(debate_id)).first()
    if debate:
        debate.participant_count = (debate.participant_count or 0) + 1
        debate.last_activity_at = datetime.utcnow()
    
    db.commit()
    db.refresh(participant)
    
    return participant

def get_debate_participants(
    db: Session,
    debate_id: str,
    include_performance_stats: bool = False
) -> List[DebateParticipant]:
    """Get all participants for a debate"""
    
    query = db.query(DebateParticipant).filter(
        DebateParticipant.debate_id == uuid.UUID(debate_id)
    )
    
    if include_performance_stats:
        # Could add joins to performance tables here
        pass
    
    return query.order_by(DebateParticipant.joined_at).all()

def update_participant_activity(
    db: Session,
    participant_id: str,
    activity_data: Dict[str, Any]
) -> Optional[DebateParticipant]:
    """Update participant activity metrics"""
    
    participant = db.query(DebateParticipant).filter(
        DebateParticipant.id == uuid.UUID(participant_id)
    ).first()
    
    if not participant:
        return None
    
    participant.last_active_at = datetime.utcnow()
    
    # Update specific metrics
    if "messages_sent" in activity_data:
        participant.messages_sent = activity_data["messages_sent"]
    if "evidence_provided" in activity_data:
        participant.evidence_provided = activity_data["evidence_provided"]
    if "challenges_issued" in activity_data:
        participant.challenges_issued = activity_data["challenges_issued"]
    if "challenges_received" in activity_data:
        participant.challenges_received = activity_data["challenges_received"]
    if "responses_given" in activity_data:
        participant.responses_given = activity_data["responses_given"]
    if "avg_confidence_score" in activity_data:
        participant.avg_confidence_score = activity_data["avg_confidence_score"]
    if "avg_response_time_seconds" in activity_data:
        participant.avg_response_time_seconds = activity_data["avg_response_time_seconds"]
    
    db.commit()
    db.refresh(participant)
    
    return participant

def update_participant_final_position(
    db: Session,
    participant_id: str,
    final_position: Dict[str, Any],
    consensus_agreement: bool,
    minority_position: Optional[Dict[str, Any]] = None
) -> Optional[DebateParticipant]:
    """Update participant's final position and consensus status"""
    
    participant = db.query(DebateParticipant).filter(
        DebateParticipant.id == uuid.UUID(participant_id)
    ).first()
    
    if not participant:
        return None
    
    participant.final_position = final_position
    participant.consensus_agreement = consensus_agreement
    participant.minority_position = minority_position
    participant.last_active_at = datetime.utcnow()
    
    db.commit()
    db.refresh(participant)
    
    return participant

# ==========================================
# DEBATE MESSAGE CRUD OPERATIONS
# ==========================================

def create_debate_message(
    db: Session,
    debate_id: str,
    sender_id: str,
    message_type: MessageType,
    content: Dict[str, Any],
    round_number: int,
    stage: DebateStage,
    evidence_sources: Optional[List[str]] = None,
    confidence_score: float = 0.8,
    recipient_id: Optional[str] = None,
    parent_message_id: Optional[str] = None
) -> DebateMessage:
    """Create a new debate message"""
    
    # Get sequence number for this round
    sequence_number = db.query(func.count(DebateMessage.id)).filter(
        and_(
            DebateMessage.debate_id == uuid.UUID(debate_id),
            DebateMessage.round_number == round_number
        )
    ).scalar() + 1
    
    message = DebateMessage(
        debate_id=uuid.UUID(debate_id),
        sender_id=uuid.UUID(sender_id),
        recipient_id=uuid.UUID(recipient_id) if recipient_id else None,
        message_type=message_type,
        round_number=round_number,
        stage=stage,
        sequence_number=sequence_number,
        content=content,
        evidence_sources=evidence_sources or [],
        confidence_score=confidence_score,
        parent_message_id=uuid.UUID(parent_message_id) if parent_message_id else None,
        created_at=datetime.utcnow()
    )
    
    db.add(message)
    
    # Update debate message count and activity
    debate = db.query(Debate).filter(Debate.id == uuid.UUID(debate_id)).first()
    if debate:
        debate.total_messages = (debate.total_messages or 0) + 1
        debate.last_activity_at = datetime.utcnow()
        
        # Update evidence count if message has evidence
        if evidence_sources:
            debate.total_evidence_items = (debate.total_evidence_items or 0) + len(evidence_sources)
    
    db.commit()
    db.refresh(message)
    
    return message

def get_debate_messages(
    db: Session,
    debate_id: str,
    round_number: Optional[int] = None,
    message_type: Optional[MessageType] = None,
    sender_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[DebateMessage]:
    """Get debate messages with filtering"""
    
    query = db.query(DebateMessage).filter(
        DebateMessage.debate_id == uuid.UUID(debate_id)
    )
    
    # Apply filters
    if round_number:
        query = query.filter(DebateMessage.round_number == round_number)
    if message_type:
        query = query.filter(DebateMessage.message_type == message_type)
    if sender_id:
        query = query.filter(DebateMessage.sender_id == uuid.UUID(sender_id))
    
    # Order by creation time and sequence
    query = query.order_by(
        DebateMessage.round_number,
        DebateMessage.sequence_number,
        DebateMessage.created_at
    )
    
    # Add pagination
    query = query.offset(offset).limit(limit)
    
    return query.all()

def update_message_analysis(
    db: Session,
    message_id: str,
    sentiment_score: Optional[float] = None,
    complexity_score: Optional[float] = None,
    evidence_quality_score: Optional[float] = None,
    responses_received: Optional[int] = None,
    challenges_generated: Optional[int] = None,
    consensus_impact: Optional[float] = None
) -> Optional[DebateMessage]:
    """Update message analysis scores"""
    
    message = db.query(DebateMessage).filter(
        DebateMessage.id == uuid.UUID(message_id)
    ).first()
    
    if not message:
        return None
    
    if sentiment_score is not None:
        message.sentiment_score = sentiment_score
    if complexity_score is not None:
        message.complexity_score = complexity_score
    if evidence_quality_score is not None:
        message.evidence_quality_score = evidence_quality_score
    if responses_received is not None:
        message.responses_received = responses_received
    if challenges_generated is not None:
        message.challenges_generated = challenges_generated
    if consensus_impact is not None:
        message.consensus_impact = consensus_impact
    
    message.processed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(message)
    
    return message

def get_message_thread(
    db: Session,
    parent_message_id: str,
    include_parent: bool = True
) -> List[DebateMessage]:
    """Get all messages in a thread (parent + responses)"""
    
    query = db.query(DebateMessage)
    
    if include_parent:
        query = query.filter(
            or_(
                DebateMessage.id == uuid.UUID(parent_message_id),
                DebateMessage.parent_message_id == uuid.UUID(parent_message_id)
            )
        )
    else:
        query = query.filter(
            DebateMessage.parent_message_id == uuid.UUID(parent_message_id)
        )
    
    return query.order_by(DebateMessage.created_at).all()

# ==========================================
# CONSENSUS CRUD OPERATIONS
# ==========================================

def create_consensus_item(
    db: Session,
    debate_id: str,
    topic: str,
    description: str,
    category: str,
    total_participants: int,
    supporting_evidence: Optional[List[Dict]] = None,
    implementation_priority: str = "medium",
    implementation_complexity: str = "moderate"
) -> ConsensusItem:
    """Create a new consensus item"""
    
    consensus_item = ConsensusItem(
        debate_id=uuid.UUID(debate_id),
        topic=topic,
        description=description,
        category=category,
        total_participants=total_participants,
        supporting_evidence=supporting_evidence or [],
        implementation_priority=implementation_priority,
        implementation_complexity=implementation_complexity,
        first_mentioned_at=datetime.utcnow()
    )
    
    db.add(consensus_item)
    db.commit()
    db.refresh(consensus_item)
    
    return consensus_item

def add_consensus_vote(
    db: Session,
    consensus_item_id: str,
    participant_id: str,
    vote: str,
    confidence: float = 0.8,
    reasoning: Optional[str] = None,
    supporting_evidence: Optional[List[Dict]] = None,
    concerns: Optional[List[Dict]] = None
) -> ConsensusVote:
    """Add vote to consensus item"""
    
    # Check if participant already voted
    existing_vote = db.query(ConsensusVote).filter(
        and_(
            ConsensusVote.consensus_item_id == uuid.UUID(consensus_item_id),
            ConsensusVote.participant_id == uuid.UUID(participant_id),
            ConsensusVote.final_vote == True
        )
    ).first()
    
    if existing_vote:
        # Mark existing vote as not final
        existing_vote.final_vote = False
    
    # Create new vote
    new_vote = ConsensusVote(
        consensus_item_id=uuid.UUID(consensus_item_id),
        participant_id=uuid.UUID(participant_id),
        vote=vote,
        confidence=confidence,
        reasoning=reasoning,
        supporting_evidence=supporting_evidence or [],
        concerns=concerns or [],
        voted_at=datetime.utcnow(),
        changed_vote=existing_vote is not None,
        final_vote=True
    )
    
    db.add(new_vote)
    
    # Update consensus item vote counts
    consensus_item = db.query(ConsensusItem).filter(
        ConsensusItem.id == uuid.UUID(consensus_item_id)
    ).first()
    
    if consensus_item:
        # Recalculate vote counts
        votes = db.query(ConsensusVote).filter(
            and_(
                ConsensusVote.consensus_item_id == uuid.UUID(consensus_item_id),
                ConsensusVote.final_vote == True
            )
        ).all()
        
        support_count = sum(1 for v in votes if v.vote == "support")
        oppose_count = sum(1 for v in votes if v.vote == "oppose")
        neutral_count = sum(1 for v in votes if v.vote == "neutral")
        
        consensus_item.support_count = support_count
        consensus_item.oppose_count = oppose_count
        consensus_item.neutral_count = neutral_count
        
        # Calculate agreement percentage
        total_votes = len(votes)
        if total_votes > 0:
            consensus_item.agreement_percentage = (support_count / total_votes) * 100
            
            # Determine consensus strength
            if consensus_item.agreement_percentage >= 80:
                consensus_item.consensus_strength = "strong"
            elif consensus_item.agreement_percentage >= 60:
                consensus_item.consensus_strength = "moderate"
            else:
                consensus_item.consensus_strength = "weak"
    
    db.commit()
    db.refresh(new_vote)
    
    return new_vote

def get_debate_consensus_items(
    db: Session,
    debate_id: str,
    category: Optional[str] = None,
    min_agreement_percentage: float = 0.0
) -> List[ConsensusItem]:
    """Get consensus items for a debate"""
    
    query = db.query(ConsensusItem).filter(
        ConsensusItem.debate_id == uuid.UUID(debate_id)
    )
    
    if category:
        query = query.filter(ConsensusItem.category == category)
    
    if min_agreement_percentage > 0:
        query = query.filter(
            ConsensusItem.agreement_percentage >= min_agreement_percentage
        )
    
    # Include votes
    query = query.options(selectinload(ConsensusItem.votes))
    
    return query.order_by(desc(ConsensusItem.agreement_percentage)).all()

def update_consensus_item_status(
    db: Session,
    consensus_item_id: str,
    consensus_reached: bool,
    expected_impact: Optional[str] = None
) -> Optional[ConsensusItem]:
    """Update consensus item status"""
    
    consensus_item = db.query(ConsensusItem).filter(
        ConsensusItem.id == uuid.UUID(consensus_item_id)
    ).first()
    
    if not consensus_item:
        return None
    
    if consensus_reached:
        consensus_item.consensus_reached_at = datetime.utcnow()
    
    if expected_impact:
        consensus_item.expected_impact = expected_impact
    
    db.commit()
    db.refresh(consensus_item)
    
    return consensus_item

# ==========================================
# ANALYTICS CRUD OPERATIONS
# ==========================================

def create_debate_analytics(
    db: Session,
    debate_id: str,
    analytics_data: Dict[str, Any]
) -> DebateAnalytics:
    """Create analytics record for debate"""
    
    analytics = DebateAnalytics(
        debate_id=uuid.UUID(debate_id),
        **analytics_data
    )
    
    db.add(analytics)
    db.commit()
    db.refresh(analytics)
    
    return analytics

def get_debate_analytics(db: Session, debate_id: str) -> Optional[DebateAnalytics]:
    """Get analytics for a debate"""
    
    return db.query(DebateAnalytics).filter(
        DebateAnalytics.debate_id == uuid.UUID(debate_id)
    ).first()

def get_user_debate_analytics(
    db: Session,
    user_id: int,
    days: int = 30
) -> Dict[str, Any]:
    """Get aggregated analytics for user's debates"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get debates with analytics
    debates_with_analytics = db.query(Debate, DebateAnalytics).join(
        DebateAnalytics, Debate.id == DebateAnalytics.debate_id
    ).filter(
        and_(
            Debate.user_id == user_id,
            Debate.created_at >= cutoff_date,
            Debate.status == DebateStatus.COMPLETED
        )
    ).all()
    
    if not debates_with_analytics:
        return {
            "total_debates": 0,
            "avg_quality_score": 0.0,
            "avg_duration_minutes": 0.0,
            "total_consensus_items": 0,
            "avg_consensus_agreement": 0.0
        }
    
    # Calculate aggregated metrics
    total_debates = len(debates_with_analytics)
    
    quality_scores = []
    durations = []
    consensus_agreements = []
    total_consensus_items = 0
    
    for debate, analytics in debates_with_analytics:
        quality_score = calculate_debate_quality_score(debate)
        quality_scores.append(quality_score)
        
        if debate.duration_seconds:
            durations.append(debate.duration_seconds / 60)  # Convert to minutes
        
        # Get consensus items for this debate
        consensus_items = get_debate_consensus_items(db, str(debate.id))
        total_consensus_items += len(consensus_items)
        
        if consensus_items:
            avg_agreement = sum(
                item.agreement_percentage or 0 for item in consensus_items
            ) / len(consensus_items)
            consensus_agreements.append(avg_agreement)
    
    return {
        "total_debates": total_debates,
        "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        "avg_duration_minutes": sum(durations) / len(durations) if durations else 0.0,
        "total_consensus_items": total_consensus_items,
        "avg_consensus_agreement": sum(consensus_agreements) / len(consensus_agreements) if consensus_agreements else 0.0,
        "period_days": days
    }

# ==========================================
# TEMPLATE CRUD OPERATIONS
# ==========================================

def create_debate_template(
    db: Session,
    name: str,
    description: str,
    category: str,
    query_template: str,
    recommended_agents: List[str],
    created_by: int,
    default_rounds: int = 3,
    urgency_level: str = "medium",
    required_parameters: Optional[List[str]] = None,
    optional_parameters: Optional[List[str]] = None,
    validation_rules: Optional[Dict[str, Any]] = None,
    is_public: bool = True
) -> DebateTemplate:
    """Create a new debate template"""
    
    template = DebateTemplate(
        name=name,
        description=description,
        category=category,
        query_template=query_template,
        recommended_agents=recommended_agents,
        created_by=created_by,
        default_rounds=default_rounds,
        urgency_level=urgency_level,
        required_parameters=required_parameters or [],
        optional_parameters=optional_parameters or [],
        validation_rules=validation_rules or {},
        is_public=is_public,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(template)
    db.commit()
    db.refresh(template)
    
    return template

def get_debate_templates(
    db: Session,
    category: Optional[str] = None,
    created_by: Optional[int] = None,
    is_public: bool = True,
    limit: int = 50,
    offset: int = 0
) -> List[DebateTemplate]:
    """Get debate templates with filtering"""
    
    query = db.query(DebateTemplate).filter(DebateTemplate.is_active == True)
    
    if is_public:
        if created_by:
            # Show public templates OR user's private templates
            query = query.filter(
                or_(
                    DebateTemplate.is_public == True,
                    DebateTemplate.created_by == created_by
                )
            )
        else:
            # Show only public templates
            query = query.filter(DebateTemplate.is_public == True)
    elif created_by:
        # Show only user's templates
        query = query.filter(DebateTemplate.created_by == created_by)
    
    if category:
        query = query.filter(DebateTemplate.category == category)
    
    # Order by usage and rating
    query = query.order_by(
        desc(DebateTemplate.avg_user_rating),
        desc(DebateTemplate.usage_count)
    )
    
    query = query.offset(offset).limit(limit)
    
    return query.all()

def update_template_usage_stats(
    db: Session,
    template_id: str,
    success: bool,
    user_rating: Optional[float] = None
) -> Optional[DebateTemplate]:
    """Update template usage statistics"""
    
    template = db.query(DebateTemplate).filter(
        DebateTemplate.id == uuid.UUID(template_id)
    ).first()
    
    if not template:
        return None
    
    # Update usage count
    template.usage_count = (template.usage_count or 0) + 1
    
    # Update success rate
    if success:
        current_successes = (template.avg_success_rate or 0.0) * ((template.usage_count or 1) - 1)
        template.avg_success_rate = (current_successes + 1.0) / template.usage_count
    
    # Update user rating
    if user_rating is not None:
        if template.avg_user_rating:
            # Calculate running average
            total_ratings = template.usage_count - 1  # Subtract 1 because we just incremented
            current_total = template.avg_user_rating * total_ratings
            template.avg_user_rating = (current_total + user_rating) / template.usage_count
        else:
            template.avg_user_rating = user_rating
    
    template.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(template)
    
    return template

# ==========================================
# AGENT PERFORMANCE CRUD OPERATIONS
# ==========================================

def create_agent_performance_record(
    db: Session,
    agent_id: str,
    agent_name: str,
    agent_version: str,
    period_start: datetime,
    period_end: datetime,
    performance_data: Dict[str, Any]
) -> AgentPerformanceHistory:
    """Create agent performance history record"""
    
    performance = AgentPerformanceHistory(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_version=agent_version,
        period_start=period_start,
        period_end=period_end,
        **performance_data
    )
    
    db.add(performance)
    db.commit()
    db.refresh(performance)
    
    return performance

def get_agent_performance_history(
    db: Session,
    agent_id: str,
    days: int = 30,
    limit: int = 10
) -> List[AgentPerformanceHistory]:
    """Get agent performance history"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    query = db.query(AgentPerformanceHistory).filter(
        and_(
            AgentPerformanceHistory.agent_id == agent_id,
            AgentPerformanceHistory.period_start >= cutoff_date
        )
    ).order_by(desc(AgentPerformanceHistory.period_start)).limit(limit)
    
    return query.all()

def get_top_performing_agents(
    db: Session,
    metric: str = "accuracy_score",
    days: int = 30,
    limit: int = 10
) -> List[Tuple[str, float]]:
    """Get top performing agents by metric"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Create query based on metric
    if metric == "accuracy_score":
        metric_col = AgentPerformanceHistory.accuracy_score
    elif metric == "user_satisfaction_score":
        metric_col = AgentPerformanceHistory.user_satisfaction_score
    elif metric == "implementation_success_rate":
        metric_col = AgentPerformanceHistory.implementation_success_rate
    else:
        metric_col = AgentPerformanceHistory.accuracy_score
    
    # Get average scores for each agent
    query = db.query(
        AgentPerformanceHistory.agent_id,
        func.avg(metric_col).label("avg_score")
    ).filter(
        AgentPerformanceHistory.period_start >= cutoff_date
    ).group_by(
        AgentPerformanceHistory.agent_id
    ).order_by(
        desc("avg_score")
    ).limit(limit)
    
    return query.all()

# ==========================================
# ADVANCED QUERY OPERATIONS
# ==========================================

def search_debates(
    db: Session,
    query_text: str,
    user_id: Optional[int] = None,
    status_filter: Optional[List[DebateStatus]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    min_confidence: Optional[float] = None,
    limit: int = 20,
    offset: int = 0
) -> List[Debate]:
    """Search debates by query text and filters"""
    
    query = db.query(Debate)
    
    # Text search (simple implementation - could use full-text search)
    if query_text:
        search_filter = or_(
            Debate.query.ilike(f"%{query_text}%"),
            Debate.description.ilike(f"%{query_text}%")
        )
        query = query.filter(search_filter)
    
    # Apply filters
    if user_id:
        query = query.filter(Debate.user_id == user_id)
    
    if status_filter:
        query = query.filter(Debate.status.in_(status_filter))
    
    if date_from:
        query = query.filter(Debate.created_at >= date_from)
    
    if date_to:
        query = query.filter(Debate.created_at <= date_to)
    
    if min_confidence:
        query = query.filter(Debate.confidence_score >= min_confidence)
    
    # Order by relevance (created_at for now)
    query = query.order_by(desc(Debate.created_at))
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    # Eager load related data
    query = query.options(
        joinedload(Debate.participants),
        joinedload(Debate.analytics)
    )
    
    return query.all()

def get_debate_statistics(
    db: Session,
    user_id: Optional[int] = None,
    days: int = 30
) -> Dict[str, Any]:
    """Get comprehensive debate statistics"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    base_query = db.query(Debate).filter(Debate.created_at >= cutoff_date)
    
    if user_id:
        base_query = base_query.filter(Debate.user_id == user_id)
    
    # Basic counts
    total_debates = base_query.count()
    completed_debates = base_query.filter(Debate.status == DebateStatus.COMPLETED).count()
    active_debates = base_query.filter(Debate.status == DebateStatus.ACTIVE).count()
    
    if total_debates == 0:
        return {
            "total_debates": 0,
            "completed_debates": 0,
            "active_debates": 0,
            "success_rate": 0.0,
            "avg_duration_minutes": 0.0,
            "avg_confidence_score": 0.0,
            "avg_participants": 0.0,
            "most_common_agents": [],
            "consensus_rate": 0.0
        }
    
    # Success rate
    success_rate = (completed_debates / total_debates) * 100
    
    # Duration statistics
    completed_debates_with_duration = base_query.filter(
        and_(
            Debate.status == DebateStatus.COMPLETED,
            Debate.duration_seconds.isnot(None)
        )
    ).all()
    
    avg_duration_minutes = 0.0
    if completed_debates_with_duration:
        total_duration = sum(d.duration_seconds for d in completed_debates_with_duration)
        avg_duration_minutes = (total_duration / len(completed_debates_with_duration)) / 60
    
    # Confidence statistics
    debates_with_confidence = base_query.filter(
        Debate.confidence_score.isnot(None)
    ).all()
    
    avg_confidence = 0.0
    if debates_with_confidence:
        avg_confidence = sum(d.confidence_score for d in debates_with_confidence) / len(debates_with_confidence)
    
    # Participant statistics
    avg_participants = 0.0
    if total_debates > 0:
        total_participants = sum(d.participant_count or 0 for d in base_query.all())
        avg_participants = total_participants / total_debates
    
    # Most common agents
    agent_usage = db.query(
        DebateParticipant.agent_id,
        func.count(DebateParticipant.id).label('usage_count')
    ).join(
        Debate, DebateParticipant.debate_id == Debate.id
    ).filter(
        Debate.created_at >= cutoff_date
    )
    
    if user_id:
        agent_usage = agent_usage.filter(Debate.user_id == user_id)
    
    most_common_agents = agent_usage.group_by(
        DebateParticipant.agent_id
    ).order_by(
        desc('usage_count')
    ).limit(5).all()
    
    # Consensus rate
    consensus_debates = base_query.filter(
        Debate.consensus_type.in_([ConsensusType.UNANIMOUS, ConsensusType.MAJORITY])
    ).count()
    
    consensus_rate = (consensus_debates / completed_debates * 100) if completed_debates > 0 else 0.0
    
    return {
        "total_debates": total_debates,
        "completed_debates": completed_debates,
        "active_debates": active_debates,
        "success_rate": success_rate,
        "avg_duration_minutes": avg_duration_minutes,
        "avg_confidence_score": avg_confidence,
        "avg_participants": avg_participants,
        "most_common_agents": [{"agent_id": agent, "count": count} for agent, count in most_common_agents],
        "consensus_rate": consensus_rate,
        "period_days": days
    }

def get_agent_collaboration_matrix(
    db: Session,
    days: int = 30,
    min_debates: int = 3
) -> Dict[str, Dict[str, float]]:
    """Get agent collaboration effectiveness matrix"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get debates with multiple participants
    debates_query = db.query(Debate).filter(
        and_(
            Debate.created_at >= cutoff_date,
            Debate.status == DebateStatus.COMPLETED,
            Debate.participant_count >= 2
        )
    )
    
    collaboration_matrix = {}
    
    for debate in debates_query:
        participants = get_debate_participants(db, str(debate.id))
        
        if len(participants) < 2:
            continue
        
        # Calculate pairwise collaboration scores
        for i, agent1 in enumerate(participants):
            for agent2 in participants[i+1:]:
                pair_key = f"{agent1.agent_id}_{agent2.agent_id}"
                reverse_key = f"{agent2.agent_id}_{agent1.agent_id}"
                
                # Use debate quality as collaboration effectiveness proxy
                quality_score = calculate_debate_quality_score(debate)
                
                if pair_key not in collaboration_matrix:
                    collaboration_matrix[pair_key] = []
                    
                collaboration_matrix[pair_key].append(quality_score)
    
    # Calculate average collaboration scores
    result_matrix = {}
    for pair_key, scores in collaboration_matrix.items():
        if len(scores) >= min_debates:  # Only include pairs with sufficient data
            agents = pair_key.split('_')
            agent1, agent2 = agents[0], agents[1]
            
            if agent1 not in result_matrix:
                result_matrix[agent1] = {}
            if agent2 not in result_matrix:
                result_matrix[agent2] = {}
            
            avg_score = sum(scores) / len(scores)
            result_matrix[agent1][agent2] = avg_score
            result_matrix[agent2][agent1] = avg_score  # Symmetric
    
    return result_matrix

def cleanup_old_debates(
    db: Session,
    days_to_keep: int = 90,
    keep_high_quality: bool = True,
    quality_threshold: float = 0.8
) -> int:
    """Clean up old debates to manage database size"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    
    # Base query for old debates
    old_debates_query = db.query(Debate).filter(
        Debate.created_at < cutoff_date
    )
    
    # Optionally preserve high-quality debates
    if keep_high_quality:
        old_debates_query = old_debates_query.filter(
            or_(
                Debate.confidence_score < quality_threshold,
                Debate.confidence_score.is_(None)
            )
        )
    
    # Only clean up completed or cancelled debates
    old_debates_query = old_debates_query.filter(
        Debate.status.in_([DebateStatus.COMPLETED, DebateStatus.CANCELLED, DebateStatus.ERROR])
    )
    
    old_debates = old_debates_query.all()
    deleted_count = len(old_debates)
    
    # Delete debates (cascade will handle related records)
    for debate in old_debates:
        db.delete(debate)
    
    db.commit()
    
    return deleted_count

# ==========================================
# BULK OPERATIONS
# ==========================================

def bulk_create_participants(
    db: Session,
    debate_id: str,
    participants_data: List[Dict[str, Any]]
) -> List[DebateParticipant]:
    """Bulk create debate participants"""
    
    participants = []
    
    for participant_data in participants_data:
        participant = DebateParticipant(
            debate_id=uuid.UUID(debate_id),
            agent_id=participant_data["agent_id"],
            agent_name=participant_data["agent_name"],
            agent_type=participant_data["agent_type"],
            agent_specialization=participant_data["agent_specialization"],
            role=participant_data.get("role", "participant"),
            joined_at=datetime.utcnow(),
            last_active_at=datetime.utcnow()
        )
        participants.append(participant)
    
    db.add_all(participants)
    
    # Update debate participant count
    debate = db.query(Debate).filter(Debate.id == uuid.UUID(debate_id)).first()
    if debate:
        debate.participant_count = len(participants_data)
        debate.last_activity_at = datetime.utcnow()
    
    db.commit()
    
    # Refresh all participants
    for participant in participants:
        db.refresh(participant)
    
    return participants

def bulk_update_participant_metrics(
    db: Session,
    participant_updates: List[Dict[str, Any]]
) -> int:
    """Bulk update participant performance metrics"""
    
    updated_count = 0
    
    for update_data in participant_updates:
        participant_id = update_data.pop("participant_id")
        
        result = db.query(DebateParticipant).filter(
            DebateParticipant.id == uuid.UUID(participant_id)
        ).update(update_data)
        
        if result:
            updated_count += 1
    
    db.commit()
    
    return updated_count

def bulk_analyze_messages(
    db: Session,
    message_analyses: List[Dict[str, Any]]
) -> int:
    """Bulk update message analysis scores"""
    
    updated_count = 0
    
    for analysis_data in message_analyses:
        message_id = analysis_data.pop("message_id")
        
        # Add processed timestamp
        analysis_data["processed_at"] = datetime.utcnow()
        
        result = db.query(DebateMessage).filter(
            DebateMessage.id == uuid.UUID(message_id)
        ).update(analysis_data)
        
        if result:
            updated_count += 1
    
    db.commit()
    
    return updated_count

# ==========================================
# EXPORT/IMPORT OPERATIONS
# ==========================================

def export_debate_data(
    db: Session,
    debate_id: str,
    include_messages: bool = True,
    include_analytics: bool = True,
    include_consensus: bool = True
) -> Optional[Dict[str, Any]]:
    """Export complete debate data for analysis or backup"""
    
    debate = get_debate(db, debate_id, include_messages=False, include_analytics=include_analytics)
    
    if not debate:
        return None
    
    export_data = {
        "debate": debate.to_dict(),
        "participants": [p.to_dict() for p in get_debate_participants(db, debate_id)],
        "export_timestamp": datetime.utcnow().isoformat(),
        "export_version": "1.0"
    }
    
    if include_messages:
        messages = get_debate_messages(db, debate_id, limit=1000)
        export_data["messages"] = [m.to_dict() for m in messages]
    
    if include_analytics and debate.analytics:
        export_data["analytics"] = debate.analytics.to_dict()
    
    if include_consensus:
        consensus_items = get_debate_consensus_items(db, debate_id)
        export_data["consensus_items"] = []
        
        for item in consensus_items:
            item_data = item.to_dict()
            # Include votes for each consensus item
            item_data["votes"] = [vote.to_dict() for vote in item.votes]
            export_data["consensus_items"].append(item_data)
    
    return export_data

def get_debate_performance_report(
    db: Session,
    debate_id: str
) -> Optional[Dict[str, Any]]:
    """Generate comprehensive performance report for a debate"""
    
    debate = get_debate(db, debate_id, include_messages=False, include_analytics=True)
    
    if not debate or debate.status != DebateStatus.COMPLETED:
        return None
    
    participants = get_debate_participants(db, debate_id, include_performance_stats=True)
    messages = get_debate_messages(db, debate_id)
    consensus_items = get_debate_consensus_items(db, debate_id)
    
    # Calculate performance metrics
    total_messages = len(messages)
    evidence_messages = [m for m in messages if m.evidence_sources and len(m.evidence_sources) > 0]
    
    report = {
        "debate_summary": {
            "id": str(debate.id),
            "query": debate.query,
            "status": debate.status.value,
            "duration_minutes": debate.duration_seconds / 60 if debate.duration_seconds else 0,
            "participants": len(participants),
            "consensus_type": debate.consensus_type.value if debate.consensus_type else None,
            "confidence_score": debate.confidence_score,
            "quality_score": calculate_debate_quality_score(debate)
        },
        
        "participation_analysis": {
            "total_messages": total_messages,
            "messages_with_evidence": len(evidence_messages),
            "evidence_rate": len(evidence_messages) / max(total_messages, 1),
            "participant_breakdown": []
        },
        
        "consensus_analysis": {
            "total_consensus_items": len(consensus_items),
            "strong_consensus_items": len([c for c in consensus_items if c.consensus_strength == "strong"]),
            "consensus_categories": {}
        },
        
        "message_flow_analysis": {
            "rounds_completed": debate.current_round,
            "avg_messages_per_round": total_messages / max(debate.current_round, 1),
            "message_types_breakdown": {}
        }
    }
    
    # Participant breakdown
    for participant in participants:
        participant_messages = [m for m in messages if str(m.sender_id) == str(participant.id)]
        
        report["participation_analysis"]["participant_breakdown"].append({
            "agent_id": participant.agent_id,
            "agent_name": participant.agent_name,
            "messages_sent": len(participant_messages),
            "evidence_provided": participant.evidence_provided,
            "avg_confidence": participant.avg_confidence_score,
            "consensus_agreement": participant.consensus_agreement,
            "participation_score": len(participant_messages) / max(total_messages, 1)
        })
    
    # Consensus categories analysis
    for item in consensus_items:
        category = item.category
        if category not in report["consensus_analysis"]["consensus_categories"]:
            report["consensus_analysis"]["consensus_categories"][category] = {
                "count": 0,
                "avg_agreement": 0.0,
                "items": []
            }
        
        report["consensus_analysis"]["consensus_categories"][category]["count"] += 1
        report["consensus_analysis"]["consensus_categories"][category]["items"].append({
            "topic": item.topic,
            "agreement_percentage": item.agreement_percentage,
            "consensus_strength": item.consensus_strength
        })
    
    # Calculate average agreement by category
    for category_data in report["consensus_analysis"]["consensus_categories"].values():
        if category_data["items"]:
            category_data["avg_agreement"] = sum(
                item["agreement_percentage"] or 0 for item in category_data["items"]
            ) / len(category_data["items"])
    
    # Message types breakdown
    message_types = {}
    for message in messages:
        msg_type = message.message_type.value
        if msg_type not in message_types:
            message_types[msg_type] = 0
        message_types[msg_type] += 1
    
    report["message_flow_analysis"]["message_types_breakdown"] = message_types
    
    # Include analytics if available
    if debate.analytics:
        report["advanced_analytics"] = debate.analytics.to_dict()
    
    return report

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def validate_debate_state_transition(
    current_status: DebateStatus,
    new_status: DebateStatus
) -> bool:
    """Validate if a debate status transition is allowed"""
    
    valid_transitions = {
        DebateStatus.PENDING: [DebateStatus.ACTIVE, DebateStatus.CANCELLED],
        DebateStatus.ACTIVE: [DebateStatus.PAUSED, DebateStatus.COMPLETED, DebateStatus.CANCELLED, DebateStatus.ERROR],
        DebateStatus.PAUSED: [DebateStatus.ACTIVE, DebateStatus.CANCELLED],
        DebateStatus.COMPLETED: [],  # Terminal state
        DebateStatus.CANCELLED: [],  # Terminal state
        DebateStatus.ERROR: [DebateStatus.ACTIVE, DebateStatus.CANCELLED]  # Can recover from error
    }
    
    return new_status in valid_transitions.get(current_status, [])

def calculate_debate_health_score(debate: Debate, participants: List[DebateParticipant]) -> float:
    """Calculate a health score for an active debate"""
    
    if debate.status != DebateStatus.ACTIVE:
        return 0.0
    
    health_factors = []
    
    # Participation health (are all participants active?)
    if participants:
        active_participants = sum(
            1 for p in participants 
            if (datetime.utcnow() - p.last_active_at).total_seconds() < 300  # Active within 5 minutes
        )
        participation_health = active_participants / len(participants)
        health_factors.append(participation_health)
    
    # Duration health (is debate taking too long?)
    if debate.started_at:
        elapsed_minutes = (datetime.utcnow() - debate.started_at).total_seconds() / 60
        max_expected_minutes = debate.max_duration_seconds / 60
        duration_health = max(0.0, 1.0 - (elapsed_minutes / max_expected_minutes))
        health_factors.append(duration_health)
    
    # Message flow health (are messages being exchanged?)
    recent_message_count = len([
        m for m in debate.messages 
        if (datetime.utcnow() - m.created_at).total_seconds() < 600  # Messages in last 10 minutes
    ]) if hasattr(debate, 'messages') else 0
    
    message_health = min(1.0, recent_message_count / 3.0)  # Expect at least 3 messages per 10 minutes
    health_factors.append(message_health)
    
    # Overall health is average of factors
    return sum(health_factors) / len(health_factors) if health_factors else 0.5

# Export all CRUD functions
__all__ = [
    # Debate operations
    "create_debate", "get_debate", "get_user_debates", "update_debate_status",
    "update_debate_results", "delete_debate",
    
    # Participant operations
    "add_debate_participant", "get_debate_participants", "update_participant_activity",
    "update_participant_final_position", "bulk_create_participants", "bulk_update_participant_metrics",
    
    # Message operations
    "create_debate_message", "get_debate_messages", "update_message_analysis",
    "get_message_thread", "bulk_analyze_messages",
    
    # Consensus operations
    "create_consensus_item", "add_consensus_vote", "get_debate_consensus_items",
    "update_consensus_item_status",
    
    # Analytics operations
    "create_debate_analytics", "get_debate_analytics", "get_user_debate_analytics",
    
    # Template operations
    "create_debate_template", "get_debate_templates", "update_template_usage_stats",
    
    # Performance operations
    "create_agent_performance_record", "get_agent_performance_history", "get_top_performing_agents",
    
    # Advanced operations
    "search_debates", "get_debate_statistics", "get_agent_collaboration_matrix",
    "cleanup_old_debates", "export_debate_data", "get_debate_performance_report",
    
    # Utility functions
    "validate_debate_state_transition", "calculate_debate_health_score",
    
    # Enums and filters
    "DebateQueryFilter"
]