'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Star, Quote, ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

interface Experience {
  id: string;
  name: string;
  title?: string;
  content: string;
  rating?: number;
  createdAt: string;
  user?: {
    name?: string;
  };
}

interface ExperienceDisplayProps {
  initialLimit?: number;
  showPagination?: boolean;
  compact?: boolean;
}

export default function ExperienceDisplay({ 
  initialLimit = 6, 
  showPagination = true,
  compact = false 
}: ExperienceDisplayProps) {
  const [experiences, setExperiences] = useState<Experience[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [limit] = useState(initialLimit);

  const fetchExperiences = async (newOffset: number = 0, showLoading: boolean = true) => {
    if (showLoading) {
      setLoading(true);
    } else {
      setLoadingMore(true);
    }

    try {
      const response = await fetch(
        `/api/experiences?limit=${limit}&offset=${newOffset}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch experiences');
      }

      const data = await response.json();

      if (newOffset === 0) {
        setExperiences(data.experiences);
      } else {
        setExperiences(prev => [...prev, ...data.experiences]);
      }

      setTotalCount(data.totalCount);
      setHasMore(data.hasMore);
      setOffset(newOffset);

    } catch (error) {
      console.error('Error fetching experiences:', error);
      toast.error('Failed to load experiences');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  };

  useEffect(() => {
    fetchExperiences();
  }, []);

  const handleLoadMore = () => {
    fetchExperiences(offset + limit, false);
  };

  const handleRefresh = () => {
    fetchExperiences(0, true);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const renderStars = (rating: number) => {
    return (
      <div className="flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`w-4 h-4 ${
              star <= rating
                ? 'text-yellow-400 fill-current'
                : 'text-gray-300'
            }`}
          />
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="w-full">
        <div className="text-center py-12">
          <RefreshCw className="w-8 h-8 text-[#5B2655] animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading experiences...</p>
        </div>
      </div>
    );
  }

  if (experiences.length === 0) {
    return (
      <div className="text-center py-12">
        <Quote className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-700 mb-2">No Experiences Yet</h3>
        <p className="text-gray-500">Be the first to share your experience with ThinkLife!</p>
      </div>
    );
  }

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-[#5B2655] mb-2">Community Experiences</h2>
          <p className="text-gray-600">
            Hear from our community of {totalCount} member{totalCount !== 1 ? 's' : ''}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          className="border-[#5B2655] text-[#5B2655] hover:bg-[#5B2655] hover:text-white"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Experiences Grid */}
      <div className={`grid gap-6 ${
        compact 
          ? 'grid-cols-1 md:grid-cols-2' 
          : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
      }`}>
        {experiences.map((experience) => (
          <Card
            key={experience.id}
            className="hover:shadow-lg transition-all duration-300 border-[#D4C4B8]/50 hover:border-[#5B2655]/30 group"
          >
            <CardContent className="p-6">
              {/* Quote Icon */}
              <div className="flex items-start gap-3">
                <Quote className="w-6 h-6 text-[#5B2655]/60 flex-shrink-0 mt-1" />
                <div className="flex-1 min-w-0">
                  {/* Title */}
                  {experience.title && (
                    <h3 className="font-semibold text-[#5B2655] mb-2 line-clamp-2">
                      {experience.title}
                    </h3>
                  )}

                  {/* Content */}
                  <p className={`text-gray-700 leading-relaxed ${
                    compact ? 'line-clamp-3' : 'line-clamp-4'
                  } mb-4`}>
                    {experience.content}
                  </p>

                  {/* Rating */}
                  {experience.rating && (
                    <div className="mb-4">
                      {renderStars(experience.rating)}
                    </div>
                  )}

                  {/* Author and Date */}
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 bg-gradient-to-br from-[#5B2655] to-[#8B5A8F] rounded-full flex items-center justify-center">
                        <span className="text-white text-xs font-medium">
                          {experience.name.charAt(0).toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <p className="font-medium text-gray-800">{experience.name}</p>
                        <p className="text-gray-500 text-xs">{formatDate(experience.createdAt)}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Load More / Pagination */}
      {showPagination && hasMore && (
        <div className="text-center">
          <Button
            onClick={handleLoadMore}
            disabled={loadingMore}
            variant="outline"
            className="border-[#5B2655] text-[#5B2655] hover:bg-[#5B2655] hover:text-white px-8"
          >
            {loadingMore ? (
              <div className="flex items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                Loading...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4" />
                Load More Stories
              </div>
            )}
          </Button>
        </div>
      )}

      {/* Footer Info */}
      {!hasMore && experiences.length > 0 && (
        <div className="text-center py-4">
          <p className="text-gray-500 text-sm">
            Showing all {experiences.length} experience{experiences.length !== 1 ? 's' : ''}
          </p>
        </div>
      )}
    </div>
  );
} 